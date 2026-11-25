/* SPDX-License-Identifier: GPL-2.0 */
/* PS-only admission. The only writes here are leased EMIO and HP0 AFI. */
#define FZ_AFI_WIDTH_64 0x3b1U
#define FZ_AFI_WIDTH_128 0x3b0U
#define D4_BIT_NS 100000000ULL
#define D4_FRAME_NS 900000000ULL
#define D4_GATE_NS 7000000000ULL
#define D4_FIRST_HPM_NS 50000000ULL

static const u8 d4_identity[D4_ID_BYTES] = {FZ_ASR_D4_BUILD_BYTES};
static const unsigned int d4_rates[D4_CLOCKS] = {
	25000000U, FZ_ASR_BOARD_CORE_MHZ * 1000000U, FZ_ASR_BOARD_CORE_MHZ * 500000U,
	FZ_ASR_D4_DSP_MHZ * 1000000U
};

static bool fz_of_board_reset(const struct fz_of_lease *lease, u32 high)
{
	return (readl(lease->gpio + FZ_GPIO_DATA_5) & FZ_FABRIC_RESET_MASK) == high &&
	       (readl(lease->gpio + FZ_GPIO_DIRM_5) & readl(lease->gpio + FZ_GPIO_OEN_5) &
		FZ_FABRIC_RESET_MASK) == FZ_FABRIC_RESET_MASK;
}

static bool fz_of_board_afi(const struct fz_of_lease *lease, u32 hp0)
{
	return readl(lease->afi[0]) == hp0 && readl(lease->afi[0] + 0x14) == hp0 &&
	       readl(lease->afi[1]) == FZ_AFI_WIDTH_128 && readl(lease->afi[1] + 0x14) == FZ_AFI_WIDTH_128;
}

static bool d4_lease_present(const struct fz_of_lease *lease)
{
	return lease->manager && !READ_ONCE(lease->observer->removed) &&
	       atomic_read(&fz_of_tree_revision) == lease->tree_revision &&
	       READ_ONCE(lease->manager->state) == FPGA_MGR_STATE_OPERATING;
}

static int d4_hold_reset(struct fz_of_lease *lease)
{
	int ctrl = gpiod_direction_output(lease->control_reset_guard, 0);
	int core = gpiod_direction_output(lease->reset_guard, 0);

	if (ctrl || core)
		return ctrl ?: core;
	return fz_of_board_reset(lease, 0) ? 0 : -EIO;
}

static int fz_of_board_map(struct fz_of_lease *lease)
{
	struct fz_of_observer *o = lease->observer;

	lease->gpio = ioremap(FZ_PS_GPIO_BASE, PAGE_SIZE);
	lease->afi[0] = ioremap(ASR_HP0_AFI, PAGE_SIZE);
	lease->afi[1] = ioremap(ASR_HP1_AFI, PAGE_SIZE);
	if (!lease->gpio || !lease->afi[0] || !lease->afi[1])
		return -ENOMEM;
	o->failed_stage = "d4-afi-baseline";
	if (!fz_of_board_afi(lease, FZ_AFI_WIDTH_64))
		return -ERANGE;
	o->failed_stage = "d4-gpio-acquire";
	lease->d4_ack = gpiod_get(o->dev, "d4-ack", GPIOD_IN);
	if (IS_ERR(lease->d4_ack))
		return PTR_ERR(lease->d4_ack);
	lease->d4_data = gpiod_get(o->dev, "d4-data", GPIOD_IN);
	if (IS_ERR(lease->d4_data))
		return PTR_ERR(lease->d4_data);
	lease->d4_req = gpiod_get(o->dev, "d4-req", GPIOD_OUT_LOW);
	if (IS_ERR(lease->d4_req))
		return PTR_ERR(lease->d4_req);
	o->failed_stage = "d4-pre-pcap-reset";
	lease->d4_reset_claimed = true;
	return d4_hold_reset(lease);
}

static int d4_read_bit(struct fz_of_lease *lease, int req, u64 deadline, u64 *seen, u8 *bit)
{
	int ack, data, again;

	for (;;) {
		ack = gpiod_get_value_cansleep(lease->d4_ack);
		if (ack < 0)
			return ack;
		if (ktime_get_raw_fast_ns() >= deadline)
			return -ETIMEDOUT;
		if (ack == req)
			break;
		if (!d4_lease_present(lease))
			return -ESTALE;
		usleep_range(20, 40);
	}
	data = gpiod_get_value_cansleep(lease->d4_data);
	again = gpiod_get_value_cansleep(lease->d4_data);
	ack = gpiod_get_value_cansleep(lease->d4_ack);
	*seen = ktime_get_raw_fast_ns();
	if (data < 0 || again < 0 || ack < 0 || data != again || ack != req)
		return -EIO;
	if (*seen >= deadline)
		return -ETIMEDOUT;
	*bit = data;
	return 0;
}

static int d4_read_frame(struct fz_of_lease *lease, struct d4_frame *frame)
{
	unsigned int bit;
	int req = 0;
	int ack = gpiod_get_value_cansleep(lease->d4_ack);

	if (ack)
		return ack < 0 ? ack : -EBUSY;
	memset(frame, 0, sizeof(*frame));
	for (bit = 0; bit < D4_FRAME_LEN * 8; bit++) {
		u64 started = ktime_get_raw_fast_ns(), seen, deadline;
		u8 data;
		int rc;

		if (!bit)
			frame->lo_ns = started;
		deadline = min3(started + D4_BIT_NS, frame->lo_ns + D4_FRAME_NS, lease->d4_gate_deadline_ns);
		if (started >= deadline)
			return -ETIMEDOUT;
		req ^= 1;
		gpiod_set_raw_value_cansleep(lease->d4_req, req);
		rc = d4_read_bit(lease, req, deadline, &seen, &data);
		if (rc)
			return rc;
		if (seen < started)
			return -EIO;
		if (!bit)
			frame->hi_ns = seen;
		frame->bytes[bit / 8] |= data << (bit % 8);
	}
	return 0;
}

static int d4_capture(struct fz_of_lease *lease, unsigned int index,
		      u32 gpio_high, unsigned int status)
{
	struct fz_of_observer *o = lease->observer;
	struct d4_frame *f = &o->d4_frame[index];
	int rc;

	if (!d4_lease_present(lease) || !fz_of_board_reset(lease, gpio_high))
		return -ESTALE;
	rc = d4_read_frame(lease, f);
	if (rc)
		return rc;
	/* Archive a full rejected frame as well. Partial transfers never publish. */
	smp_store_release(&o->d4_frames, index + 1);
	if (!d4_frame_ok(f, FZ_ASR_D4_PROFILE_ID, d4_identity, status) ||
	    (index && !d4_next(&o->d4_frame[index - 1], f)) ||
	    (!index && !d4_get32(f->bytes + 4)))
		return -EBADMSG;
	return d4_lease_present(lease) && fz_of_board_reset(lease, gpio_high) ? 0 : -ESTALE;
}

static int fz_of_board_prepare(struct fz_of_lease *lease)
{
	struct fz_of_observer *o = lease->observer;
	u64 target, now;
	int rc;

	/* Full PCAP firmware can toggle GPIO. Start the verified hold again here. */
	lease->d4_gate_deadline_ns = ktime_get_raw_fast_ns() + D4_GATE_NS;
	o->failed_stage = "d4-post-pcap-reset";
	rc = d4_hold_reset(lease);
	if (rc)
		return rc;
	gpiod_set_raw_value_cansleep(lease->d4_req, 0);
	msleep(2100);
	o->failed_stage = "d4-held-capture-a";
	rc = d4_capture(lease, 0, 0, 0);
	if (rc)
		return rc;
	target = o->d4_frame[0].hi_ns + NSEC_PER_SEC;
	now = ktime_get_raw_fast_ns();
	if (now < target)
		msleep(DIV_ROUND_UP(target - now, NSEC_PER_MSEC));
	o->failed_stage = "d4-held-capture-b";
	rc = d4_capture(lease, 1, 0, 0);
	if (rc)
		return rc;
	o->failed_stage = "d4-clock-rates";
	if (!d4_check_rates(&o->d4_frame[0], &o->d4_frame[1], d4_rates))
		return -ERANGE;
	o->failed_stage = "d4-afi-width";
	if (!fz_of_board_reset(lease, 0) || !fz_of_board_afi(lease, FZ_AFI_WIDTH_64))
		return -ERANGE;
	writel(FZ_AFI_WIDTH_128, lease->afi[0]);
	if (readl(lease->afi[0]) != FZ_AFI_WIDTH_128)
		return -EIO;
	writel(FZ_AFI_WIDTH_128, lease->afi[0] + 0x14);
	if (!fz_of_board_afi(lease, FZ_AFI_WIDTH_128))
		return -EIO;
	usleep_range(100, 200);
	o->failed_stage = "d4-release-core-request";
	if (!d4_lease_present(lease) || !fz_of_legacy_idle() || !fz_of_board_reset(lease, 0))
		return -ESTALE;
	gpiod_set_raw_value_cansleep(lease->reset_guard, 1);
	usleep_range(100, 200);
	rc = d4_capture(lease, 2, BIT(31), 1);
	if (rc)
		return rc;
	o->failed_stage = "d4-release-control-request";
	gpiod_set_raw_value_cansleep(lease->control_reset_guard, 1);
	usleep_range(100, 200);
	rc = d4_capture(lease, 3, FZ_FABRIC_RESET_MASK, 0xbf);
	if (rc)
		return rc;
	o->failed_stage = "d4-final-ps-gate";
	if (!fz_of_legacy_idle() || !fz_of_board_afi(lease, FZ_AFI_WIDTH_128))
		return -ESTALE;
	lease->d4_admitted_ns = ktime_get_raw_fast_ns();
	if (lease->d4_admitted_ns >= lease->d4_gate_deadline_ns)
		return -ETIMEDOUT;
	lease->board_ready = true;
	return 0;
}

static bool d4_first_hpm_ready(const struct fz_of_lease *lease)
{
	u64 now = ktime_get_raw_fast_ns();

	return lease->board_ready && d4_lease_present(lease) &&
	       now >= lease->d4_admitted_ns && now - lease->d4_admitted_ns <= D4_FIRST_HPM_NS &&
	       now < lease->d4_gate_deadline_ns && fz_of_board_reset(lease, FZ_FABRIC_RESET_MASK);
}

