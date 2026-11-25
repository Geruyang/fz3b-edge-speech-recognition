print("BEGIN retire",flush=True)
import json, os, subprocess
from pathlib import Path

def cmd(*argv):
    return subprocess.check_output(argv, universal_newlines=True)
def report(tag, value):
    print(json.dumps([tag, value]), flush=True)
def stop(name):
    name += ".service"
    result = cmd("systemctl", "stop", name)
    status = dict(line.split("=",1) for line in cmd("systemctl","show","-p","ActiveState","-p","MainPID","-p","Result","-p","ExecMainStatus",name).splitlines())
    report(name,status)
    assert status["ActiveState"] == "inactive" and status["MainPID"] == "0" and status["Result"] == "success", status
def no_users(paths, names=()):
    found = []
    for pr in Path("/proc").glob("[0-9]*"):
        try:
            if any(s in (pr/"cmdline").read_bytes() for s in names):
                found.append([pr.name,"client"])
            for fd in (pr/"fd").iterdir():
                try: target = os.readlink(str(fd))
                except FileNotFoundError: continue
                if target in paths: found.append([pr.name, target])
            for line in (pr/"maps").read_text(errors="replace").splitlines():
                if any(p in line for p in paths): found.append([pr.name, line])
        except (FileNotFoundError,ProcessLookupError):
            continue
    report("device_users",found)
    assert not found
assert Path("/proc/sys/kernel/random/boot_id").read_text().strip()=="BUILD_BOOT_UUID"
loader=json.loads(cmd("python3","-B","/root/workspace/network10-v10.5-pl-loader/fz3b_pl_loader.py","status"))
assert loader["ok"] and loader["result"]["status"]=="READY_A0005"
cpuunit="fz3b-app-cpu-clock.service"
inv=cmd("systemctl","show","-p","InvocationID","--value",cpuunit).strip()
logs=cmd("journalctl","-b","-o","cat","--no-pager","_SYSTEMD_INVOCATION_ID="+inv)
values=[json.loads(s)["initial_raw"] for s in logs.splitlines() if s.startswith("{") and '"initial_raw"' in s]
assert len(values)==1, "missing original CPU-clock snapshot in journal"
report("cpu_clock_before_service",values[0])
remote=sorted(Path("/sys/class/remoteproc").glob("remoteproc*"))
def r5():
    return [[str(x.resolve()),(x/"state").read_text().strip(),(x/"firmware").read_text().strip()] for x in remote]
before=r5()
report("r5_before",before)
assert len(remote)==2 and all(any(("app-r5"+str(i)) in path and fw=="fz3b-app-r5"+str(i)+".elf" and state=="running" for path,state,fw in before) for i in (0,1))
stop("fz3b-vision-console")
no_users(["/dev/network10_v10_signed40"],(b"benchmark_cnn_pipeline_v10",b"ort_partition_runner",b"run_cnn_pipeline_v10"))
stop("fz3b-app-cpu-clock")
raw=cmd("/root/fz3b-vision-console-20261009-001/read_clock")
report("cpu_clock_released",raw)
assert raw == values[0]
no_users(["/dev/fz3b_cpu_clock"])
module=Path("/sys/module/fz3b_cpu_clock")
if module.exists():
    assert (module/"refcnt").read_text().strip()=="0" and not list((module/"holders").iterdir())
    cmd("/sbin/rmmod","fz3b_cpu_clock")
assert not module.exists() and not Path("/dev/fz3b_cpu_clock").exists()
stop("fz3b-app-r5")
after=r5()
report("r5_after", after)
assert all(item[1]=="offline" for item in after) and not Path("/run/fz3b-r5-ready.json").exists()

print("BEGIN factory",flush=True)
import json, os, subprocess
from pathlib import Path

def run(*args): return subprocess.check_output(args,universal_newlines=True).strip()
def report(k,v): print(json.dumps([k,v]),flush=True)
loader="/root/workspace/network10-v10.5-pl-loader/fz3b_pl_loader.py"
def call(action):
    reply=json.loads(run("python3","-B",loader,action))
    report(action,reply)
    assert reply["ok"]
    return reply["result"]
assert Path("/proc/sys/kernel/random/boot_id").read_text().strip()=="BUILD_BOOT_UUID"
for svc in ("fz3b-vision-console","fz3b-app-cpu-clock","fz3b-app-r5"):
    assert run("systemctl","show","--value","-p","ActiveState",svc+".service")=="inactive"
assert all((r/"state").read_text().strip()=="offline" for r in Path("/sys/class/remoteproc").glob("remoteproc*"))
owners=[]
for pid in Path("/proc").glob("[0-9]*"):
    try:
        for fd in (pid/"fd").iterdir():
            try: target=os.readlink(str(fd))
            except FileNotFoundError: continue
            if target.startswith(("/dev/network","/dev/uio","/dev/mem","/dev/ttyS")): owners.append([pid.name,target])
        for line in (pid/"maps").read_text(errors="replace").splitlines():
            if any(fragment in line for fragment in ("/dev/network","/dev/uio","/dev/mem","/dev/ttyS")): owners.append([pid.name,line])
    except (FileNotFoundError,ProcessLookupError): continue
report("owners",owners)
assert not owners
prev=call("status")
assert prev["ready"] and prev["status"]=="READY_A0005" and prev["observed"]["detected_target"]=="a0005"
factory=call("switch-factory")
assert factory["status"]=="READY_FACTORY" and factory["ready"] and factory["target"]=="factory" and not factory["failure"]
obs=factory["observed"]
assert obs["detected_target"]=="factory" and obs["manager_state"]=="operating"
raw=dict(prev["observed"]["raw_clocks"])
raw["pl1_ref_ctrl"]=0x52000
assert obs["raw_clocks"]==raw
for mod in ("cnn_ddr_v10_signed40_candidate","fz3b_pl1_iopll_lease"):
    assert not (Path("/sys/module")/mod).exists()
assert not Path("/sys/kernel/config/device-tree/overlays/network10-v10.5-a0005").exists()
assert Path("/sys/module/fpgadrv").exists()
for name in ("network_latency","network_throughput"):
    assert (Path("/dev")/name).is_char_device()

print("BEGIN oldowners",flush=True)
import fcntl,json,os,subprocess
from pathlib import Path
def cmd(*args): return subprocess.check_output(args,universal_newlines=True).strip()
def emit(k,v): print(json.dumps([k,v]),flush=True)
unit="fz3b-pl-loader.service"
loader="/root/workspace/network10-v10.5-pl-loader/fz3b_pl_loader.py"
check=json.loads(cmd("python3","-B",loader,"status"))
assert check["ok"] and check["result"]["status"]=="READY_FACTORY" and check["result"]["observed"]["detected_target"]=="factory"
assert not Path("/sys/module/fz3b_pl1_iopll_lease").exists()
assert cmd("systemctl","show","-p","RefuseManualStop","--value",unit)=="yes"
drop=Path("/run/systemd/system/fz3b-pl-loader.service.d/90-t016-maintenance.conf")
drop.parent.mkdir(exist_ok=True)
with drop.open("x") as f: f.write("[Unit]\nRefuseManualStop=no\n")
try:
    cmd("systemctl","daemon-reload")
    assert cmd("systemctl","show","-p","RefuseManualStop","--value",unit)=="no"
    cmd("systemctl","stop",unit)
    states=cmd("systemctl","show","-p","MainPID","-p","ActiveState","-p","Result",unit)
    emit("loader_after_stop",states)
    assert {"MainPID=0","ActiveState=inactive","Result=success"} <= set(states.splitlines())
finally:
    assert drop.read_text()=="[Unit]\nRefuseManualStop=no\n"
    drop.unlink()
    cmd("systemctl","daemon-reload")
    assert cmd("systemctl","show","-p","RefuseManualStop","--value",unit)=="yes"
with open("/run/lock/fz3b-pl-loader.lock","a") as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    paths=("/dev/fpgadrv","/dev/network_latency","/dev/network_throughput","/dev/fz3b_pl1_iopll_lease","/dev/uio","/dev/mem")
    users=[]
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            for fd in (proc/"fd").iterdir():
                try: target=os.readlink(str(fd))
                except FileNotFoundError: continue
                if target.startswith(paths): users.append([proc.name,target])
            for line in (proc/"maps").read_text(errors="replace").splitlines():
                if any(p in line for p in paths): users.append([proc.name,line])
        except (FileNotFoundError,ProcessLookupError): continue
    emit("old_owners",users)
    mod=Path("/sys/module/fpgadrv")
    ref=(mod/"refcnt").read_text().strip()
    holders=list((mod/"holders").iterdir())
    emit("fpgadrv",[ref,[s.name for s in holders]])
    assert not users and ref=="0" and not holders
    cmd("/sbin/rmmod","fpgadrv")
    assert not mod.exists() and not Path("/sys/class/fpgadrv").exists() and not list(Path("/dev").glob("fpgadrv*")) and not Path("/sys/bus/platform/devices/fpga-driver@0/driver").exists()
    emit("factory_driver_unloaded",True)
    emit("handoff_verified",True)
