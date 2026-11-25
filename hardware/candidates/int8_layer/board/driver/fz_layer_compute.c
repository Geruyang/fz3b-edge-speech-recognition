// SPDX-License-Identifier: GPL-2.0
#include <linux/module.h>
#include <linux/platform_device.h>
#include <linux/of_platform.h>
#include <linux/of_address.h>
#include <linux/of_device.h>
#include <linux/of_irq.h>
#include <linux/fpga/fpga-mgr.h>
#include <linux/dma-mapping.h>
#include <linux/interrupt.h>
#include <linux/miscdevice.h>
#include <linux/firmware.h>
#include <linux/gpio/consumer.h>
#include <linux/delay.h>
#include <linux/kthread.h>
#include <linux/completion.h>
#include <linux/uaccess.h>
#include <crypto/hash.h>
#include "layer_job.h"
#include "frozen.h"
#include "d4_qualify.h"

#define FZ_FABRIC_RESET_MASK (BIT(30) | BIT(31))
#define FZ_PS_GPIO_BASE 0xff0a0000U
#define FZ_GPIO_DATA_5 0x054U
#define FZ_GPIO_DIRM_5 0x344U
#define FZ_GPIO_OEN_5 0x348U
#define ASR_HP0_AFI 0xfd380000U
#define ASR_HP1_AFI 0xfd390000U
#define FZ_ASR_BOARD_CORE_MHZ 200U
#define FZ_ASR_D4_DSP_MHZ 200U
#define FZ_ASR_D4_PROFILE_ID 17U
#define GUARD 64
#define PREFIX "fzspeech/int8-layer-request/"

static atomic_t fz_of_tree_revision = ATOMIC_INIT(0);
struct fz_of_observer {
    struct device *dev;
    const char *failed_stage;
    bool removed;
    unsigned int d4_frames;
    struct d4_frame d4_frame[D4_CAPTURE_COUNT];
};
struct fz_of_lease {
    struct fz_of_observer *observer;
    struct fpga_manager *manager;
    struct gpio_desc *reset_guard, *control_reset_guard, *d4_req, *d4_ack, *d4_data;
    void __iomem *gpio, *afi[2];
    int tree_revision;
    bool d4_reset_claimed, board_ready;
    u64 d4_gate_deadline_ns, d4_admitted_ns;
};
/* Board clock/reset sequences and locked-owner check from the tested D4 window.
 * The frozen single-core status has lane1 tied low, all other bits are identical. */
#include "legacy_idle.inc"
#include "single_clock.inc"

struct slab { void *v; dma_addr_t addr; size_t bytes; };
static struct slab weights[10], params[10], data[4];
static struct fz_of_observer observer;
static struct fz_of_lease lease = { .observer = &observer };
static struct platform_device *pdev;
static void __iomem *regs;
static struct miscdevice node;
static DEFINE_MUTEX(job_lock);
static DECLARE_COMPLETION(done);
static atomic64_t opened = ATOMIC64_INIT(0), armed = ATOMIC64_INIT(0);
static bool fault;
static u64 irq_ns, seq;
static const int rows[] = {16,16,8,8,4,4,2,2,8,8};
static size_t data_size[] = {LAYER_BOUNDARY,LAYER_HISTORY,LAYER_BOUNDARY,LAYER_HISTORY};
static int slab_alloc(struct slab *s, size_t n)
{
    s->bytes = n;
    s->v = dma_alloc_coherent(&pdev->dev, n + 2*GUARD, &s->addr, GFP_KERNEL);
    if (!s->v) return -ENOMEM;
    if ((s->addr & 15) || (u64)s->addr + n + 2*GUARD > BIT_ULL(30))
        return -ERANGE;
    memset(s->v, 0xa5, n + 2*GUARD);
    return 0;
}
static bool good_guard(const struct slab *s)
{
    return !memchr_inv(s->v, 0xa5, GUARD) && !memchr_inv(s->v+GUARD+s->bytes,0xa5,GUARD);
}
static int verify(const struct firmware *f, size_t n, const u8 *expected)
{
    struct crypto_shash *tfm;
    struct shash_desc *desc;
    u8 sum[32];
    int rc;
    if (f->size != n) return -EBADMSG;
    tfm = crypto_alloc_shash("sha256",0,0);
    if (IS_ERR(tfm)) return PTR_ERR(tfm);
    desc = kzalloc(sizeof(*desc)+crypto_shash_descsize(tfm),GFP_KERNEL);
    if (!desc) { crypto_free_shash(tfm); return -ENOMEM; }
    desc->tfm=tfm;
    rc=crypto_shash_digest(desc,f->data,n,sum);
    kfree(desc); crypto_free_shash(tfm);
    return rc ? rc : (memcmp(sum,expected,32) ? -EBADMSG : 0);
}
static int load_slab(struct slab *s,const struct layer_asset *a)
{
    const struct firmware *f;
    int rc=request_firmware_direct(&f,a->name,&pdev->dev);
    if (rc) return rc;
    rc=verify(f,a->bytes,a->sha);
    if (!rc) rc=slab_alloc(s,a->bytes);
    if (!rc) memcpy(s->v+GUARD,f->data,a->bytes);
    release_firmware(f);
    return rc;
}
static irqreturn_t complete_irq(int irq,void *opaque)
{
    u32 status=readl(regs+12);
    if (!(status&1)) return IRQ_NONE;
    WRITE_ONCE(irq_ns,ktime_get_raw_fast_ns());
    writel(1,regs+12);
    readl(regs+12);
    if (atomic64_cmpxchg(&armed,1,0)==1) complete(&done);
    else WRITE_ONCE(fault,true);
    return IRQ_HANDLED;
}
static void pointer_reg(unsigned int base,struct slab *s)
{
    writel(lower_32_bits(s->addr+GUARD),regs+base);
    writel(0,regs+base+4);
}
static long submit(struct file *file,unsigned int cmd,unsigned long arg)
{
    struct layer_job j;
    u64 begin,prepared,start,awake,completed;
    int i,rc=0;
    if (cmd!=LAYER_RUN) return -ENOTTY;
    if (copy_from_user(&j,(void __user *)arg,sizeof(j))) return -EFAULT;
    if (j.layer>=10 || !j.batch || j.batch>2) return -EINVAL;
    mutex_lock(&job_lock);
    if (fault || !lease.board_ready || !d4_lease_present(&lease)) {rc=-EIO;goto out;}
    begin=ktime_get_raw_fast_ns();
    if (copy_from_user(data[0].v+GUARD,u64_to_user_ptr(j.input),LAYER_BOUNDARY) ||
        copy_from_user(data[1].v+GUARD,u64_to_user_ptr(j.history),LAYER_HISTORY)) {rc=-EFAULT;goto out;}
    memset(data[2].v+GUARD,0xa5,LAYER_BOUNDARY);
    memset(data[3].v+GUARD,0xa5,LAYER_HISTORY);
    dma_wmb();
    prepared=ktime_get_raw_fast_ns();
    if (!(readl(regs)&4) || readl(regs+12)!=0 || atomic64_read(&armed)) {rc=-EBUSY;goto quarantine;}
    pointer_reg(24,&weights[j.layer]); pointer_reg(36,&params[j.layer]);
    for (i=0;i<4;i++) pointer_reg(48+12*i,&data[i]);
    writel(rows[j.layer],regs+96); writel(j.batch,regs+104);
    reinit_completion(&done); atomic64_set(&armed,1);
    j.sequence=++seq;
    start=ktime_get_raw_fast_ns();
    writel(1,regs);
    if (!wait_for_completion_timeout(&done,msecs_to_jiffies(3000))) {rc=-ETIMEDOUT;goto quarantine;}
    awake=ktime_get_raw_fast_ns(); completed=READ_ONCE(irq_ns);
    j.control=readl(regs); j.result=readl(regs+16);
    if (fault || (j.control&6)!=6 || j.result || readl(regs+12) || completed<start || awake<completed) {
        rc=-EIO;goto quarantine;
    }
    dma_rmb();
    for (i=0;i<4;i++) if (!good_guard(&data[i])) {rc=-EOVERFLOW;goto quarantine;}
    if (!good_guard(&weights[j.layer]) || !good_guard(&params[j.layer])) {rc=-EOVERFLOW;goto quarantine;}
    if (copy_to_user(u64_to_user_ptr(j.output),data[2].v+GUARD,LAYER_BOUNDARY) ||
        copy_to_user(u64_to_user_ptr(j.next_history),data[3].v+GUARD,LAYER_HISTORY)) {rc=-EFAULT;goto out;}
    j.copy_in_ns=prepared-begin; j.setup_ns=start-prepared; j.start_irq_ns=completed-start;
    j.irq_wake_ns=awake-completed; j.copy_out_ns=ktime_get_raw_fast_ns()-awake;
    if (copy_to_user((void __user *)arg,&j,sizeof(j))) rc=-EFAULT;
    goto out;
quarantine:
    WRITE_ONCE(fault,true);
    dev_err(&pdev->dev,"isolated rc=%d seq=%llu layer=%u; retaining DMA buffers\n",rc,seq,j.layer);
out:
    mutex_unlock(&job_lock);
    return rc;
}
static int layer_open(struct inode *inode,struct file *file)
{ return atomic64_cmpxchg(&opened,0,1) ? -EBUSY : 0; }
static int layer_close(struct inode *inode,struct file *file)
{ atomic64_set(&opened,0); return 0; }
static const struct file_operations ops={
    .owner=THIS_MODULE,.open=layer_open,.release=layer_close,.unlocked_ioctl=submit,.llseek=no_llseek
};

static int maintain(void *unused)
{
    struct device *dev=&pdev->dev;
    struct device_node *m;
    struct fpga_image_info config={0};
    const struct firmware *fw=NULL;
    struct resource *control;
    unsigned int i;
    int irq,rc;
    u64 bus,cpu,size,alloc_start=ktime_get_raw_fast_ns();
    observer.dev=dev; observer.failed_stage="of-dma-range";
    rc=of_dma_get_range(dev->of_node,&bus,&cpu,&size);
    if (rc) goto failed;
    if (bus || cpu || size!=BIT_ULL(30) || !dev->dma_mask ||
        (*dev->dma_mask&DMA_BIT_MASK(30))!=DMA_BIT_MASK(30) ||
        (dev->coherent_dma_mask&DMA_BIT_MASK(30))!=DMA_BIT_MASK(30) ||
        dev->dma_pfn_offset || !get_dma_ops(dev) || of_dma_is_coherent(dev->of_node)) {rc=-ERANGE;goto failed;}
    observer.failed_stage="dma-mask";
    rc=dma_set_mask_and_coherent(dev,DMA_BIT_MASK(30));
    if (rc) goto failed;
    if (*dev->dma_mask!=DMA_BIT_MASK(30) || dev->coherent_dma_mask!=DMA_BIT_MASK(30)) {rc=-ERANGE;goto failed;}
    dev_info(dev,"DMA domain verified: low 1 GiB identity mapping, coherent allocations\n");
    observer.failed_stage="frozen-model-banks";
    for (i=0;i<10;i++) {
        rc=load_slab(&weights[i],&banks[i][0]); if (rc) goto failed;
        rc=load_slab(&params[i],&banks[i][1]); if (rc) goto failed;
    }
    for (i=0;i<4;i++) {rc=slab_alloc(&data[i],data_size[i]); if (rc) goto failed;}
    dev_info(dev,"verified resident assets and allocated coherent buffers in %llu ns\n",ktime_get_raw_fast_ns()-alloc_start);
    observer.failed_stage="control-resource";
    control=platform_get_resource(pdev,IORESOURCE_MEM,0);
    if (!control || control->start!=0xa0010000 || resource_size(control)!=0x10000) {rc=-EINVAL;goto failed;}
    if (!request_mem_region(control->start,resource_size(control),"fz-layer-live")) {rc=-EBUSY;goto failed;}
    regs=ioremap(control->start,resource_size(control));
    if (!regs) {rc=-ENOMEM;goto failed;}
    observer.failed_stage="manager";
    if (!fz_of_legacy_idle()) {rc=-EBUSY;goto failed;}
    m=of_parse_phandle(dev->of_node,"fpga-mgr",0);
    if (!m || !of_device_is_compatible(m,"xlnx,zynqmp-pcap-fpga")) {of_node_put(m);rc=-ENODEV;goto failed;}
    lease.manager=of_fpga_mgr_get(m); of_node_put(m);
    if (IS_ERR_OR_NULL(lease.manager)) {rc=-EBUSY;goto failed;}
    lease.reset_guard=gpiod_get(dev,"fabric-reset-guard",GPIOD_ASIS);
    if (IS_ERR(lease.reset_guard)) {rc=PTR_ERR(lease.reset_guard);goto failed;}
    lease.control_reset_guard=gpiod_get(dev,"control-reset-guard",GPIOD_ASIS);
    if (IS_ERR(lease.control_reset_guard)) {rc=PTR_ERR(lease.control_reset_guard);goto failed;}
    rc=request_firmware_direct(&fw,image.name,dev); if (rc) goto failed;
    rc=verify(fw,image.bytes,image.sha); if (rc) goto failed;
    rc=fz_of_board_map(&lease); if (rc) goto failed;
    observer.failed_stage="pcap";
    rc=fpga_mgr_buf_load(lease.manager,&config,(const char *)fw->data,fw->size);
    release_firmware(fw); fw=NULL;
    if (rc) goto failed;
    rc=fz_of_board_prepare(&lease); if (rc) goto failed;
    observer.failed_stage="first-idle";
    if (!d4_first_hpm_ready(&lease)) {rc=-ESTALE;goto failed;}
    if (!(readl(regs)&4) || readl(regs+12)) {rc=-EIO;goto failed;}
    observer.failed_stage="irq";
    irq=platform_get_irq(pdev,0);
    if (irq<=0) {rc=-ENXIO;goto failed;}
    rc=request_irq(irq,complete_irq,0,"fz-layer-live",&lease); if (rc) goto failed;
    writel(1,regs+4); writel(1,regs+8);
    if (readl(regs+4)!=1 || readl(regs+8)!=1) {rc=-EIO;goto failed;}
    node.minor=MISC_DYNAMIC_MINOR; node.name="fz-layer-live"; node.fops=&ops; node.mode=0600;
    rc=misc_register(&node); if (rc) goto failed;
    dev_info(dev,"ready: fixed 200 MHz single core Linux IRQ %d\n",irq);
    goto hold;
failed:
    if (fw) release_firmware(fw);
    WRITE_ONCE(fault,true);
    dev_err(dev,"stopped at %s rc=%d; resources retained\n",observer.failed_stage,rc);
hold:
    for (i=0;i<observer.d4_frames;i++)
        dev_info(dev,"frame %u %llu %llu %*phN\n",i,observer.d4_frame[i].lo_ns,
                 observer.d4_frame[i].hi_ns,48,observer.d4_frame[i].bytes);
    /* Keep manager and DMA ownership in this task until normal board reboot. */
    while (!kthread_should_stop()) schedule_timeout_interruptible(HZ*30);
    return 0;
}
static int layer_probe(struct platform_device *platform)
{
    struct device_node *expected=of_find_node_by_path("/fz-layer-dma-bus/layer-core@a0010000");
    struct task_struct *task;
    bool match=expected && platform->dev.of_node==expected;
    of_node_put(expected);
    if (pdev || !match) return -ENODEV;
    dev_info(&platform->dev,"fz-layer: bound exact layer node\n");
    pdev=platform;
    task=kthread_run(maintain,NULL,"fz_int8_owner");
    if (IS_ERR(task)) {pdev=NULL;return PTR_ERR(task);}
    return 0;
}
static int parent_probe(struct platform_device *platform)
{
    struct device_node *expected=of_find_node_by_path("/fz-layer-dma-bus");
    bool match=expected && platform->dev.of_node==expected;
    int rc;
    of_node_put(expected);
    if (!match || of_get_child_count(platform->dev.of_node)!=1) return -ENODEV;
    rc=of_platform_populate(platform->dev.of_node,NULL,NULL,&platform->dev);
    dev_info(&platform->dev,"fz-layer: populated single child rc=%d\n",rc);
    return rc;
}
static const struct of_device_id core_ids[]={{.compatible="edgespeech,int8-layer-request-v2"},{}};
static const struct of_device_id bus_ids[]={{.compatible="edgespeech,int8-layer-dma-bus"},{}};
static struct platform_driver core_driver={
    .probe=layer_probe,.driver={.name="fz-layer-compute",.of_match_table=core_ids,.suppress_bind_attrs=true}
};
static struct platform_driver bus_driver={
    .probe=parent_probe,.driver={.name="fz-layer-owner",.of_match_table=bus_ids,.suppress_bind_attrs=true}
};
static int __init layer_init(void)
{
    int rc=platform_driver_register(&core_driver);
    if (rc) return rc;
    rc=platform_driver_register(&bus_driver);
    if (rc) platform_driver_unregister(&core_driver);
    return rc;
}
module_init(layer_init);
/* The supervised benchmarking module has no runtime unload path. */
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("FZ3B bounded fixed-model INT8 layer measurement");
