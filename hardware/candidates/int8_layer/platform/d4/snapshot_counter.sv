`timescale 1ps/1ps

module d4_snapshot_counter (
    input  wire clk,
    input  wire request_token,
    output logic acknowledged_token = 1'b0,
    output logic [31:0] frozen_count = '0
);
    (* ASYNC_REG = "TRUE", SHREG_EXTRACT = "NO" *) logic [1:0] request_sync = '0;
    logic [31:0] edge_count = '0;

    always @(posedge clk) begin
        request_sync <= {request_sync[0], request_token};
        edge_count <= edge_count + 1'b1;
        if (request_sync[1] != acknowledged_token) begin
            frozen_count <= edge_count;
            acknowledged_token <= request_sync[1];
        end
    end
endmodule
