# Implement only a fully resolved platform that fits the project resource envelope.
set input [file normalize [lindex $argv 0]]
set out [file normalize [lindex $argv 1]]
set fp [open $input/utilization.rpt]; set util [read $fp]; close $fp
foreach {label cap} {{CLB LUTs\*} 60000 {CLB Registers} 110000 {Block RAM Tile} 180 {DSPs} 320} {
    set pattern [format {\| %s\s+\|\s*([0-9.]+)\s*\|} $label]
    if {![regexp $pattern $util -> actual] || $actual>$cap} {error "Platform outside budget: $label"}
}
file mkdir $out
set resume [expr {[lindex $argv 2] eq "placed"}]
set incremental [expr {[lindex $argv 2] eq "incremental"}]
if {$incremental && ($argc!=4 || ![file isfile [lindex $argv 3]])} {error "Missing routed reference"}
if {$resume} {open_checkpoint $out/placed.dcp} else {open_checkpoint $input/platform.dcp}
set_param general.maxThreads [expr {$incremental ? 4 : 2}]
if {[llength [get_cells -hier -quiet -filter {IS_BLACKBOX == 1}]]} {error "unresolved hardware"}
set mmcm [get_cells -hier -filter {REF_NAME == MMCME4_ADV}]
if {[llength $mmcm]!=1} {error "missing unique D4 clock manager"}
if {!$resume} {
    set_property LOC MMCM_X0Y2 $mmcm
    opt_design
    if {$incremental} {
        set ref [file normalize [lindex $argv 3]]
        puts "LAYOUT_REFERENCE $ref"
        read_checkpoint -incremental -directive TimingClosure $ref
    }
    place_design -directive Explore
    write_checkpoint -force $out/placed.dcp
}
phys_opt_design -directive Explore
route_design -directive Explore
report_route_status -file $out/route.rpt
report_utilization -file $out/utilization.rpt
report_utilization -hierarchical -file $out/hierarchy.rpt
report_timing_summary -delay_type min_max -report_unconstrained -file $out/timing.rpt
report_clock_interaction -file $out/interactions.rpt
report_cdc -details -file $out/cdc.rpt
report_drc -file $out/drc.rpt
check_timing -verbose -file $out/check.rpt
report_clocks -file $out/clocks.rpt
write_checkpoint -force $out/routed.dcp
if {$incremental} {report_incremental_reuse -file $out/reuse.rpt}
