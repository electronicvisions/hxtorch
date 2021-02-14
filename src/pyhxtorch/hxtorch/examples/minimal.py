"""
A minimal example that multiplies a vector and a matrix.
"""
import torch
import hxtorch


def main():
    # initialize the connection, uses default nightly calib for the setup
    hxtorch.init_hardware()

    x = torch.full((128,), 10.)  # input range: 0...31
    w = torch.full((128, 256), 20.)  # weight range: -63...63
    # this weight uses the whole upper half of the chip

    out = hxtorch.matmul(
       x, w,  # the same as in `torch.matmul`
       num_sends=1,  # number of subsequent sends of the input in the same integration step
       wait_between_events=5,  # wait between sending the individual vector entries (in FPGA cycles)
    )  # output range: -128...127

    log = hxtorch.logger.get("hxtorch.examples.minimal")
    log.info(
        f"Input (mean): {x.mean()}, "
        f"weight (mean): {w.mean()}, "
        f"output (mean): {out.mean()}"
    )
    hxtorch.release_hardware()


if __name__ == "__main__":
    main()
