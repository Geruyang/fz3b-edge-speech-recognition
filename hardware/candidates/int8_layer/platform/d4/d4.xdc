set_property PACKAGE_PIN D4 [get_ports ref_d4]
set_property IOSTANDARD LVCMOS18 [get_ports ref_d4]
create_clock -name ref_d4 -period 40.000 [get_ports ref_d4]
# Same clocking-wizard proposal as the repaired D4 diagnostic design.
set_input_jitter [get_clocks ref_d4] 0.400
