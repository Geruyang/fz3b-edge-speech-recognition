`timescale 1ps/1ps

module fz_dsp2x_d4_clocking (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ref_d4 CLK",
       X_INTERFACE_PARAMETER = "XIL_INTERFACENAME ref_d4, FREQ_HZ 25000000" *)
    input  wire ref_d4,
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 clk_ref CLK",
       X_INTERFACE_PARAMETER = "XIL_INTERFACENAME clk_ref, FREQ_HZ 25000000" *)
    output wire clk_ref,
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 clk_core CLK",
       X_INTERFACE_PARAMETER = "XIL_INTERFACENAME clk_core, FREQ_HZ 200000000" *)
    output wire clk_core,
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 clk_ctrl CLK",
       X_INTERFACE_PARAMETER = "XIL_INTERFACENAME clk_ctrl, FREQ_HZ 100000000" *)
    output wire clk_ctrl,
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 clk_fast CLK",
       X_INTERFACE_PARAMETER = "XIL_INTERFACENAME clk_fast, FREQ_HZ 400000000" *)
    output wire clk_fast,
    output wire locked,
    output wire clkinstopped,
    output logic reset_req = 1'b1,
    output wire reset_asserted,
    output wire core_ready,
    output wire ctrl_ready,
    output logic [31:0] fault_epoch = '0
);
    wire ref_buffered, feedback, mmcm_core, mmcm_ctrl, mmcm_fast;
    IBUF u_ref_input (.I(ref_d4), .O(ref_buffered));
    MMCME4_ADV #(
        .BANDWIDTH("OPTIMIZED"), .COMPENSATION("AUTO"), .STARTUP_WAIT("FALSE"),
        .DIVCLK_DIVIDE(1), .CLKFBOUT_MULT_F(48.0), .CLKIN1_PERIOD(40.0),
        .CLKOUT0_DIVIDE_F(6.0), .CLKOUT0_PHASE(0.0), .CLKOUT0_DUTY_CYCLE(0.5),
        .CLKOUT1_DIVIDE(12), .CLKOUT1_PHASE(0.0), .CLKOUT1_DUTY_CYCLE(0.5),
        .CLKOUT2_DIVIDE(3), .CLKOUT2_PHASE(0.0), .CLKOUT2_DUTY_CYCLE(0.5)
    ) u_mmcm (
        .CLKIN1(ref_buffered), .CLKIN2(1'b0), .CLKINSEL(1'b1),
        .CLKFBIN(feedback), .CLKFBOUT(feedback), .CLKOUT0(mmcm_core), .CLKOUT1(mmcm_ctrl),
        .LOCKED(locked), .CLKINSTOPPED(clkinstopped), .RST(reset_req), .PWRDWN(1'b0),
        .DADDR(7'h0), .DCLK(1'b0), .DEN(1'b0), .DI(16'h0), .DWE(1'b0),
        .PSCLK(1'b0), .PSEN(1'b0), .PSINCDEC(1'b0), .CDDCREQ(1'b0),
        .CLKFBOUTB(), .CLKOUT0B(), .CLKOUT1B(), .CLKOUT2(mmcm_fast), .CLKOUT2B(),
        .CLKOUT3(), .CLKOUT3B(), .CLKOUT4(), .CLKOUT5(), .CLKOUT6(),
        .CLKFBSTOPPED(), .DO(), .DRDY(), .PSDONE(), .CDDCDONE()
    );
    BUFG u_core_buffer (.I(mmcm_core), .O(clk_core));
    BUFG u_ctrl_buffer (.I(mmcm_ctrl), .O(clk_ctrl));
    BUFG u_fast_buffer (.I(mmcm_fast), .O(clk_fast));
    BUFG u_ref_buffer (.I(ref_buffered), .O(clk_ref));

    // Capture the MMCM's direct input-stop status with a single PRE source.
    // Consume the episode once even if status remains asserted until relock.
    logic stopped_latched = 1'b0;
    logic stop_serviced = 1'b0;
    (* ASYNC_REG = "TRUE", SHREG_EXTRACT = "NO" *) logic [1:0] stopped_sync = '0;
    (* ASYNC_REG = "TRUE", SHREG_EXTRACT = "NO" *) logic [2:0] lock_qualified = '0;
    logic running = 1'b0;
    logic [2:0] reset_edges = '0;
    logic [11:0] lock_wait_edges = '0;
    wire stopped_pending = stopped_sync[1] & ~stop_serviced;

    always @(posedge clk_ref or posedge clkinstopped) begin
        if (clkinstopped) stopped_latched <= 1'b1;
        else if (reset_req || stop_serviced) stopped_latched <= 1'b0;
    end
    always @(posedge clk_ref) stopped_sync <= {stopped_sync[0], stopped_latched};

    always @(posedge clk_ref or negedge locked) begin
        if (!locked) lock_qualified <= '0;
        else lock_qualified <= {lock_qualified[1:0], 1'b1};
    end

    always @(posedge clk_ref) begin
        if (!stopped_sync[1]) stop_serviced <= 1'b0;
        if (reset_req) begin
            if (stopped_sync[1]) stop_serviced <= 1'b1;
            reset_edges <= reset_edges + 1'b1;
            lock_wait_edges <= '0;
            if (&reset_edges) reset_req <= 1'b0;
        end else if (stopped_pending || (running && !lock_qualified[2]) ||
                     (!running && !lock_qualified[2] && (&lock_wait_edges))) begin
            if (stopped_sync[1]) stop_serviced <= 1'b1;
            running <= 1'b0;
            reset_req <= 1'b1;
            reset_edges <= '0;
            fault_epoch <= fault_epoch + 1'b1;
        end else if (lock_qualified[2]) begin
            running <= 1'b1;
            lock_wait_edges <= '0;
        end else if (!running) begin
            lock_wait_edges <= lock_wait_edges + 1'b1;
        end
    end

    assign reset_asserted = !running | !lock_qualified[2] | stopped_pending | reset_req;

    (* ASYNC_REG = "TRUE", SHREG_EXTRACT = "NO" *) logic [3:0] core_release = '0;
    (* ASYNC_REG = "TRUE", SHREG_EXTRACT = "NO" *) logic [3:0] ctrl_release = '0;
    always @(posedge clk_core or negedge locked) begin
        if (!locked) core_release <= '0;
        else if (reset_asserted) core_release <= '0;
        else core_release <= {core_release[2:0], 1'b1};
    end
    always @(posedge clk_ctrl or negedge locked) begin
        if (!locked) ctrl_release <= '0;
        else if (reset_asserted) ctrl_release <= '0;
        else ctrl_release <= {ctrl_release[2:0], 1'b1};
    end
    assign core_ready = core_release[3];
    assign ctrl_ready = ctrl_release[3];
endmodule
