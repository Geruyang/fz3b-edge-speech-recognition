# Local-only export. Source is the same routed DCP as the qualified reports.
if {[version -short] ne "2026.1" || $argc != 2} {error "Expected Vivado 2026.1, checkpoint directory and output directory"}
set input [file normalize [lindex $argv 0]]
set out [file normalize [lindex $argv 1]]
file mkdir $out
cd $out
if {[file exists layer.bit] || [file exists layer.xsa]} {error "Existing export"}
open_checkpoint $input/routed.dcp
set_param general.maxThreads 2
if {[llength [get_cells -quiet -hier -filter {IS_BLACKBOX == 1}]]} {error "Unresolved hardware"}
foreach {clk period} {ref_d4 40 mmcm_core 5 mmcm_ctrl 10} {
    if {abs([get_property PERIOD [get_clocks $clk]]-$period)>0.001} {error "Unexpected $clk"}
}
foreach type {min max} {
    set path [get_timing_paths -delay_type $type -max_paths 1]
    if {[llength $path]!=1 || [get_property SLACK $path]<=0} {error "Timing not closed: $type"}
}
report_drc -ruledeck bitstream_checks -no_waivers -file bitstream-checks.rpt
set violations [get_drc_violations -quiet]
set outfp [open findings.tsv w]
foreach v $violations {
    set sev [get_property SEVERITY $v]
    puts $outfp "$sev\t$v"
    if {$sev in {"Error" "Critical Warning"}} {error "Blocking design rule: $v"}
}
close $outfp
write_bitstream layer.bit
write_hw_platform -fixed -include_bit -file layer.xsa
puts "LAYER_LOCAL_EXPORT_COMPLETE"
