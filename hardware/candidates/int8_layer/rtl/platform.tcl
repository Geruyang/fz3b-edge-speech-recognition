# Resource comparison platform. Dedicated layer IP, D4 PS shell, 3:1 shared HP0.
set root [file normalize [file join [file dirname [info script]] ../../../..]]
set ipdir [file normalize [lindex $argv 0]]
set out [file normalize [lindex $argv 1]]
file mkdir $out
set d4 $root/hardware/candidates/int8_layer/platform/d4
set part xazu3eg-sfvc784-1-i
if {$argc<3 || ![regexp {^0x[0-9a-fA-F]{24}$} [lindex $argv 2]]} {error "Missing 96-bit candidate identity"}
set resume [expr {[lindex $argv 3] eq "resume" || [lindex $argv 4] eq "resume"}]
if {$resume} {
    open_project $out/p/layer.xpr
    foreach run [get_runs -filter {IS_SYNTHESIS == 1}] {
        if {[string match -nocase {*error*} [get_property STATUS $run]]} {reset_run $run}
    }
} else {
create_project layer $out/p -part $part -force
if {[lindex $argv 3] eq "dsp300"} {config_ip_cache -disable_cache}
set_property ip_repo_paths $ipdir [current_project]
update_ip_catalog
set_property target_language Verilog [current_project]
add_files -norecurse [list $d4/clocking.sv $d4/capability.sv $d4/snapshot_counter.sv]
add_files -fileset constrs_1 -norecurse $d4/d4.xdc
create_bd_design platform
proc ip {type name} { create_bd_cell -type ip -vlnv $type $name }
proc wire {a b} { connect_bd_net [get_bd_pins $a] [get_bd_pins $b] }
proc bus {a b} { connect_bd_intf_net [get_bd_intf_pins $a] [get_bd_intf_pins $b] }
source $root/hardware/candidates/int8_layer/platform/ps_preset.tcl
set_property -dict {CONFIG.PSU__GPIO_EMIO__PERIPHERAL__ENABLE 1 CONFIG.PSU__NUM_FABRIC_RESETS 2 CONFIG.PSU__GPIO_EMIO_WIDTH 94} $ps
create_bd_cell -type module -reference fz_dsp2x_d4_clocking clocks
set d4port [create_bd_port -dir I -type clk -freq_hz 25000000 ref_d4]
connect_bd_net $d4port [get_bd_pins clocks/ref_d4]
foreach domain {ctrl core} {
    ip xilinx.com:ip:proc_sys_reset:5.0 reset_$domain
    wire clocks/clk_$domain reset_$domain/slowest_sync_clk
    wire clocks/${domain}_ready reset_$domain/dcm_locked
}
set_property CONFIG.C_AUX_RESET_HIGH 0 [get_bd_cells reset_core]
wire ps/pl_resetn1 reset_ctrl/ext_reset_in
wire ps/pl_resetn0 reset_core/ext_reset_in
wire ps/pl_resetn1 reset_core/aux_reset_in
create_bd_cell -type module -reference fz_d4_capability capability
# This synthesis-only shell has its own profile and build signature.
set_property CONFIG.PROFILE_ID 17 [get_bd_cells capability]
set_property CONFIG.BUILD_ID [lindex $argv 2] [get_bd_cells capability]
foreach {src dst} {
    clocks/clk_ref capability/clk_ref clocks/clk_core capability/clk_core
    clocks/clk_ctrl capability/clk_ctrl clocks/clk_core capability/clk_dsp
    clocks/locked capability/locked clocks/core_ready capability/provider_core_ready
    clocks/ctrl_ready capability/provider_ctrl_ready clocks/reset_asserted capability/provider_reset_asserted
    clocks/fault_epoch capability/fault_epoch ps/pl_resetn0 capability/ps_core_reset_n
    ps/pl_resetn1 capability/ps_ctrl_reset_n reset_core/peripheral_aresetn capability/core_peripheral_reset_n
    reset_ctrl/peripheral_aresetn capability/ctrl_peripheral_reset_n
    reset_ctrl/interconnect_aresetn capability/ctrl_interconnect_reset_n
    ps/emio_gpio_o capability/gpio_o ps/emio_gpio_t capability/gpio_t capability/gpio_i ps/emio_gpio_i
} { wire $src $dst }
ip xilinx.com:hls:fz_int8_layer:1.0 kernel
ip xilinx.com:ip:xlconstant:1.1 hi
ip xilinx.com:ip:xlconstant:1.1 lo
set_property CONFIG.CONST_VAL 0 [get_bd_cells lo]
wire hi/dout kernel/mac_event_full_n
wire hi/dout kernel/vector_event_full_n
wire lo/dout capability/lane1_ready
wire reset_core/peripheral_aresetn capability/lane0_ready
wire reset_core/peripheral_aresetn kernel/ap_rst_n
wire clocks/clk_core kernel/ap_clk
wire kernel/interrupt ps/pl_ps_irq0
ip xilinx.com:ip:smartconnect:1.0 control
set_property -dict {CONFIG.NUM_SI 1 CONFIG.NUM_MI 1 CONFIG.NUM_CLKS 2} [get_bd_cells control]
bus ps/M_AXI_HPM0_FPD control/S00_AXI
bus control/M00_AXI kernel/s_axi_control
wire clocks/clk_ctrl ps/maxihpm0_fpd_aclk
wire clocks/clk_ctrl control/aclk
wire clocks/clk_core control/aclk1
wire reset_ctrl/interconnect_aresetn control/aresetn
ip xilinx.com:ip:smartconnect:1.0 ddr
set_property -dict {CONFIG.NUM_SI 3 CONFIG.NUM_MI 1} [get_bd_cells ddr]
wire clocks/clk_core ddr/aclk
wire reset_core/interconnect_aresetn ddr/aresetn
foreach port {0 1 2 3} {wire clocks/clk_core ps/saxihp${port}_fpd_aclk}
set i 0
foreach name {weight coeff data} {
    bus kernel/m_axi_$name [format {ddr/S%02d_AXI} $i]
    incr i
}
bus ddr/M00_AXI ps/S_AXI_HP0_FPD
assign_bd_address -target_address_space [get_bd_addr_spaces ps/Data] -offset 0xA0010000 -range 64K [get_bd_addr_segs kernel/s_axi_control/Reg]
foreach name {weight coeff data} {
    assign_bd_address -target_address_space [get_bd_addr_spaces kernel/Data_m_axi_$name] -offset 0 -range 1G [get_bd_addr_segs ps/SAXIGP2/HP0_DDR_LOW]
}
validate_bd_design
save_bd_design
write_bd_tcl -force $out/topology.tcl
set bd [get_files platform.bd]
generate_target all $bd
add_files -norecurse [make_wrapper -files $bd -top]
set_property top platform_wrapper [current_fileset]
}
if {[lindex $argv 3] eq "dsp300" && ![llength [get_runs -quiet platform_kernel_0_synth_1]]} {
    create_ip_run [get_files platform.bd]
}
set pre [open $out/thread_cap.tcl w]
puts $pre {set_param general.maxThreads 2}
close $pre
foreach run [get_runs -filter {IS_SYNTHESIS == 1}] {set_property STEPS.SYNTH_DESIGN.TCL.PRE $out/thread_cap.tcl $run}
if {[lindex $argv 3] eq "dsp300"} {
    set kernel_run [get_runs platform_kernel_0_synth_1]
    set option STEPS.SYNTH_DESIGN.ARGS.MAX_DSP
    if {$option in [list_property $kernel_run]} {
        set_property $option 300 $kernel_run
    } else {
        set_property {STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS} {-max_dsp 300} $kernel_run
    }
}
if {$resume} {reset_run synth_1}
launch_runs synth_1 -jobs 1
wait_on_run synth_1
if {![string match {*synth_design Complete!*} [get_property STATUS [get_runs synth_1]]]} {error "platform synthesis incomplete"}
open_run synth_1
if {[llength [get_cells -quiet -hier -filter {IS_BLACKBOX == 1}]]} {error "unresolved platform logic"}
report_utilization -file $out/utilization.rpt
report_utilization -hierarchical -file $out/hierarchy.rpt
report_timing_summary -file $out/timing.rpt
write_checkpoint -force $out/platform.dcp
