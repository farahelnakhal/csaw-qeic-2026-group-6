# Writeup: Quantum Execution Integrity Challenge

Both stages are an integrity violation, with confidence 0.93. In both, the circuit, mapping and schedule are correct. The fault sits in a stale control or model record.

## Stage 1

The compiled circuit matches the intended 3-qubit GHZ circuit. It is unitary-equivalent once the qubit layout is applied. A maintenance ticket (maint_2048) moved logical q2 from physical qubit 2 to 4, and the mapping and schedule follow the move correctly.

The second CX (logical q1 to q2) now runs on physical qubits 1 and 4. Its control binding bind_73 still loads the old K19 bundle cb_73c, profile cp_017, built for qubits 1 and 2. The correct K20 profile cp_026 was never bound.

The counts show the effect. The ideal output is half 000 and half 111. The run gave about half 000 and half 011, so qubit 2 stays at 0. The distance to the ideal distribution is 0.517, against 0.066 for the earlier healthy run. Bits c0 and c1 still agree 95% of the time, while c2 agrees with them about 51% of the time. Observed 011 counts are 3756, against about 84 expected from the healthy run, a gap of 403 standard deviations.

Model selection (BIC) ranks "second CX has no effect" first. Noise as calibrated scores 26,712 worse. Noise inflated by a free amount scores 15,817 worse. The counts alone cannot separate a failed gate from a stale binding, so the provenance records make that call.

Two calibration oddities are decoys: low gate quality on node_d and high readout error on node_f. Neither qubit appears in the mapping or schedule.

Repair: rebind bind_73 to cp_026 and rerun. Before each job, check that every binding has the execution epoch and that its profile endpoints match the scheduled qubits.

## Stage 2

The compiled circuit is the intended circuit with the target gate moved from index 2346 to 694. Per-qubit order is identical. Mapping, schedule and runtime records agree and use revision D21.

The device model model-20 comes from the old D20 geometry and sets the Q1-Q4 coupling to 16 MHz. The current design design-21 says 18 MHz. The calibration cal-21 was built from model-20, so its 15.625 ns pulse gives the right angle only for 16 MHz.

Fitting cos(n*omega*t) to the eight characterization probes gives omega = 0.11310 rad/ns, which is g = 18.00 MHz and 1.125 times nominal. The 95% interval from a bootstrap is 17.997 to 18.005 MHz. The realized angle is 1.7672 rad (9 pi/16) against the intended pi/2, an over-rotation of 0.196 rad. The verification circuit agrees: its X1 value is -0.193 where about 0 is expected, which implies the same angle.

The global parity check reads exactly 1.0 and looks healthy. RZZ gates commute with parity, so it passes for any angle. A 32-qubit matrix product state run confirms this. Parity stays 1.0 with the faulty gate, while Z1 moves from 0.207 to 0.078 and Z4 from 0.381 to 0.258. Qubit 31 stays at 0.894. The run needs a bond dimension of only 19 to 23, and a tighter truncation threshold changes results by less than 1e-6.

Model selection prefers 18 MHz. The stale 16 MHz scores 651,540 worse in BIC, and 16 MHz with free decay scores 403,171 worse. These large numbers depend on an assumed probe noise of about 0.0023, so they show a wide margin and carry little precision.

Qubit 27 (short T1) and qubit 30 (high readout error) are decoys, as is the duplicate operation id op_00694 on an unrelated rz gate.

Repair: regenerate the model from design-21 at 18 MHz and rebuild cal-21 with a pulse of 13.89 ns for pi/2. Before each calibration, check that the model and design geometry fingerprints match.

## Bonus

Three optional checks sit in extensions/ and leave solve.py and result.json unchanged.

QuTiP (B3) simulates the ZZ evolution over the eight probes. Error is 1.0 at 16 MHz and 0.0010 at 18 MHz, against a tolerance of 0.004. Its own fit gives 18.0004 MHz. The as-run pulse has fidelity 0.9904 with the intended gate, and the corrected pulse gives 1.0000.

scqubits (B2) rebuilds the two transmons from model-20, with frequencies 6.08 and 6.15 GHz and anharmonicity near -280 MHz. The dressed splitting matches sqrt(delta^2 + 4g^2) for both couplings, so the reduced model is consistent.

Quantum Metal (B1) builds the 32-transmon D21 layout. Its physical order matches the backend coupling chain, and Q1 and Q4 sit 100 um apart edge to edge.

B1 and B2 check consistency only. The D20 geometry is missing from the files, so they cannot derive 16 versus 18 MHz. Only B3 confirms 18 MHz independently.

## Requirements

Both stages give a verdict, confidence, layer assessments, fault location, rejected decoys, cited evidence, repair and repair validation. The Stage 1 diagnostics use two offline simulated probes, 8192 of the 10,000 allowed shots. The Stage 2 chain covers design, model and calibration provenance, with the coupling consequence in MHz and radians. The README recommends a global MPS simulation with a convergence check, and that is done. The bonus is done as described above. The result file passes validate_submission.py.


## Notes

We used AI assistance for part of the code generation and for cleaning up this writeup. The team ran the analysis, checked the results and chose what to submit. We also had trouble pushing to the group repository, so the submission commit may appear under a different account or with unusual history. The content of solve.py and outputs/result.json is the same files we validated with validate_submission.py.

## Reproduce

```sh
pip install -r requirements-core.txt
python evaluate.py --submission solve.py --output outputs/result.json
python validate_submission.py outputs/result.json --submission solve.py
python extensions/forensics/forensics_stage1.py
python extensions/forensics/stage2_mps_convergence.py
```

The bonus scripts need requirements-advanced.txt. Run them as:

```sh
python extensions/qutip_validation/qutip_zz_validation.py
python extensions/scqubits_model/scqubits_reduced_model.py
QT_QPA_PLATFORM=offscreen python extensions/device_reconstruction/metal_geometry_reconstruction.py
```
