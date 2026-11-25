# Executed on the board in the boot-checked maintenance directory.
import ctypes as C
import fcntl
import json
import struct

folder=WD/"smoke"
assert not (folder/"issued").exists()
names=("input","old","output","next")
expect=[(folder/(n+".bin")).read_bytes() for n in names]
assert list(map(len,expect))==[8200,43272,8200,43272]
buf=[C.create_string_buffer(x,len(x)) for x in expect[:2]]
buf.extend(C.create_string_buffer(len(x)) for x in expect[2:])
fmt=struct.Struct("<II10QII")
request=bytearray(fmt.pack(0,2,*[C.addressof(x) for x in buf],*([0]*8)))
with open("/dev/fz-layer-live","rb+",buffering=0) as dev:
    (folder/"issued").write_text("1")
    fcntl.ioctl(dev,0xc0004c01 | fmt.size<<16,request,True)
r=fmt.unpack(request)
for i,name in ((2,"actual-output.bin"),(3,"actual-history.bin")):
    (folder/name).write_bytes(buf[i].raw)
summary=dict(zip(("sequence","copy_in_ns","setup_ns","start_irq_ns","irq_wake_ns","copy_out_ns","control","result"),r[6:]))
summary.update(output_equal=buf[2].raw==expect[2],history_equal=buf[3].raw==expect[3])
(folder/"smoke.json").write_text(json.dumps(summary))
print(json.dumps(summary),flush=True)
assert summary["output_equal"] and summary["history_equal"] and summary["sequence"]==1
