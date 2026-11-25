`timescale 1ps/1ps

// Four-clock inspection lane. No AXI, job or reset control interfaces.
module fz_d4_capability #(
    parameter logic [7:0] PROFILE_ID = 8'd1,
    parameter logic [95:0] BUILD_ID = 96'h33de0329c50f5e2eb79446f5,
    parameter integer CAPTURE_TICKS = 4096,
    parameter integer IDLE_TICKS = 50_000_000
) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 clk_ref CLK" *)
    input wire clk_ref,
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 clk_core CLK" *)
    input wire clk_core,
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 clk_ctrl CLK" *)
    input wire clk_ctrl,
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 clk_dsp CLK" *)
    input wire clk_dsp,
    input wire [93:0] gpio_o,
    input wire [93:0] gpio_t,
    output wire [93:0] gpio_i,
    input wire locked,
    input wire provider_core_ready,
    input wire provider_ctrl_ready,
    input wire provider_reset_asserted,
    input wire [31:0] fault_epoch,
    input wire ps_core_reset_n,
    input wire ps_ctrl_reset_n,
    input wire core_peripheral_reset_n,
    input wire ctrl_peripheral_reset_n,
    input wire ctrl_interconnect_reset_n,
    input wire lane0_ready,
    input wire lane1_ready
);
    localparam int BODY_BITS = 352;
    localparam int LAST_BIT = 383;
    logic ps_ack = 1'b0;
    logic ps_data = 1'b0;
    assign gpio_i = {87'b0, ps_data, ps_ack, 5'b0};

    (* ASYNC_REG = "TRUE", SHREG_EXTRACT = "NO" *) logic [1:0] req_sync = '0;
    (* ASYNC_REG = "TRUE", SHREG_EXTRACT = "NO" *) logic [2:0] clock_meta = '0, clock_sync = '0;
    (* ASYNC_REG = "TRUE", SHREG_EXTRACT = "NO" *) logic [7:0] status_meta = '0, status_sync = '0;
    (* ASYNC_REG = "TRUE", SHREG_EXTRACT = "NO" *) logic [3:0] dsp_release = '0;

    // This witness is clocked by the actual DSP net and shares the physical
    // core peripheral reset. It is independent of mailbox payload state.
    always @(posedge clk_dsp or negedge core_peripheral_reset_n) begin
        if (!core_peripheral_reset_n) dsp_release <= '0;
        else dsp_release <= {dsp_release[2:0], 1'b1};
    end

    always @(posedge clk_ref) begin
        req_sync <= {req_sync[0], gpio_o[4] & ~gpio_t[4]};
        clock_meta <= {provider_ctrl_ready, provider_core_ready, locked};
        clock_sync <= clock_meta;
        status_meta <= {dsp_release[3], lane1_ready, lane0_ready, ctrl_interconnect_reset_n, ctrl_peripheral_reset_n,
                        core_peripheral_reset_n, ps_ctrl_reset_n, ps_core_reset_n};
        status_sync <= status_meta;
    end

    wire qualified = !provider_reset_asserted && (&clock_sync);
    logic snapshot_token = 1'b0;
    wire ref_done, core_done, ctrl_done, dsp_done;
    wire [31:0] ref_count, core_count, ctrl_count, dsp_count;
    (* KEEP_HIERARCHY = "yes" *) d4_snapshot_counter u_ref_counter (
        .clk(clk_ref), .request_token(snapshot_token), .acknowledged_token(ref_done), .frozen_count(ref_count)
    );
    (* KEEP_HIERARCHY = "yes" *) d4_snapshot_counter u_core_counter (
        .clk(clk_core), .request_token(snapshot_token), .acknowledged_token(core_done), .frozen_count(core_count)
    );
    (* KEEP_HIERARCHY = "yes" *) d4_snapshot_counter u_ctrl_counter (
        .clk(clk_ctrl), .request_token(snapshot_token), .acknowledged_token(ctrl_done), .frozen_count(ctrl_count)
    );
    (* KEEP_HIERARCHY = "yes" *) d4_snapshot_counter u_dsp_counter (
        .clk(clk_dsp), .request_token(snapshot_token), .acknowledged_token(dsp_done), .frozen_count(dsp_count)
    );
    (* ASYNC_REG = "TRUE", SHREG_EXTRACT = "NO" *) logic [3:0] ack_meta = '0, ack_sync = '0;
    always @(posedge clk_ref) begin
        ack_meta <= {ref_done, core_done, ctrl_done, dsp_done};
        ack_sync <= ack_meta;
    end
    wire all_done = ack_sync == {4{snapshot_token}};

    typedef enum logic [2:0] { RESYNC, IDLE, WAIT_CAPTURE, COLLECT, LOAD_BIT, PUBLISH_BIT, WAIT_NEXT } state_t;
    state_t state = RESYNC;
    logic response_level = 1'b0;
    logic [31:0] frame_seq = '0;
    logic [31:0] capture_epoch = '0;
    logic [7:0] capture_status = '0;
    logic [$clog2(CAPTURE_TICKS+1)-1:0] capture_wait = '0;
    logic [$clog2(IDLE_TICKS+1)-1:0] idle_wait = '0;
    logic [BODY_BITS-1:0] frame_body = '0;
    logic [31:0] crc = '1;
    logic [8:0] bit_index = '0;
    logic frame_valid = 1'b0;
    logic frame_fault = 1'b0;
    logic final_bit_poisoned = 1'b0;
    wire fault_now = !qualified || (capture_epoch != fault_epoch) || (capture_status != status_sync);
    wire poison = frame_valid && (frame_fault || fault_now);
    wire payload_bit = (bit_index < BODY_BITS) ? frame_body[bit_index] : ~crc[bit_index[4:0]];

    task automatic freeze_frame(input logic valid, input logic timed_out, input logic fault);
        logic [7:0] flags;
        flags = {1'b0, fault, timed_out, !provider_reset_asserted, clock_sync, valid};
        frame_body <= {BUILD_ID, 24'b0, status_sync, fault_epoch,
                       (valid ? dsp_count : 32'b0), (valid ? ctrl_count : 32'b0),
                       (valid ? core_count : 32'b0), (valid ? ref_count : 32'b0),
                       frame_seq, PROFILE_ID, flags, 8'h02, 8'hD4};
        frame_valid <= valid;
        frame_fault <= 1'b0;
        final_bit_poisoned <= 1'b0;
        bit_index <= '0;
        crc <= '1;
        state <= LOAD_BIT;
    endtask

    always @(posedge clk_ref) begin
        if (frame_valid && fault_now) frame_fault <= 1'b1;
        case (state)
            RESYNC: if (!req_sync[1]) state <= IDLE;

            IDLE: if (req_sync[1] != ps_ack) begin
                response_level <= req_sync[1];
                frame_seq <= frame_seq + 1'b1;
                capture_epoch <= fault_epoch;
                capture_status <= status_sync;
                capture_wait <= '0;
                frame_valid <= 1'b0;
                state <= WAIT_CAPTURE;
                // An unresolved token cannot be reused, even after timeout.
                if (qualified && all_done) snapshot_token <= ~snapshot_token;
                else capture_wait <= CAPTURE_TICKS;
            end

            WAIT_CAPTURE: begin
                if (fault_now) freeze_frame(1'b0, 1'b0, 1'b1);
                else if (capture_wait >= CAPTURE_TICKS) freeze_frame(1'b0, 1'b1, 1'b0);
                else if (all_done) state <= COLLECT;
                else capture_wait <= capture_wait + 1'b1;
            end

            // All source buses remain frozen until their next token.
            COLLECT: freeze_frame(!fault_now, 1'b0, fault_now);

            LOAD_BIT: begin
                ps_data <= payload_bit ^ ((bit_index == LAST_BIT) && poison);
                if (bit_index < BODY_BITS)
                    crc <= (crc >> 1) ^ ((crc[0] ^ payload_bit) ? 32'hEDB88320 : 32'b0);
                final_bit_poisoned <= (bit_index == LAST_BIT) && poison;
                state <= PUBLISH_BIT;
            end

            PUBLISH_BIT: begin
                if (bit_index == LAST_BIT && poison && !final_bit_poisoned) begin
                    ps_data <= ~ps_data;
                    final_bit_poisoned <= 1'b1;
                end else begin
                    ps_ack <= response_level;
                    idle_wait <= '0;
                    if (bit_index == LAST_BIT) begin
                        frame_valid <= 1'b0;
                        state <= IDLE;
                    end else begin
                        bit_index <= bit_index + 1'b1;
                        state <= WAIT_NEXT;
                    end
                end
            end

            WAIT_NEXT: begin
                if (req_sync[1] != ps_ack) begin
                    response_level <= req_sync[1];
                    state <= LOAD_BIT;
                end else if (idle_wait >= IDLE_TICKS-1) begin
                    ps_ack <= 1'b0;
                    ps_data <= 1'b0;
                    frame_valid <= 1'b0;
                    state <= RESYNC;
                end else begin
                    idle_wait <= idle_wait + 1'b1;
                end
            end

            default: begin
                ps_ack <= 1'b0;
                frame_valid <= 1'b0;
                state <= RESYNC;
            end
        endcase
    end
endmodule
