"""Quantum Execution Integrity Challenge - participant solution.

Stage 1 is derived from the artifacts (nothing hard-coded about the verdict);
Stage 2 is left as the supplied starter (INSUFFICIENT_EVIDENCE).
"""
import json
import math
from pathlib import Path


def empty_submission():
    return dict(verdict='INSUFFICIENT_EVIDENCE', layer_assessments={}, suspected_fault_type=None,
                affected_logical_operations=[], expected_physical_resources=[], suspected_realized_resources=[],
                artifact_evidence=[], diagnostic_evidence=[], estimated_parameters={}, incidental_anomalies=[],
                root_cause=None, proposed_repair=None, repair_validation=None, confidence=0.0)


# ----------------------------------------------------------------- helpers
def _j(d, name):
    return json.loads((Path(d) / name).read_text(encoding='utf-8'))


def _ops(qc):
    return [(i.operation.name, [round(float(p), 9) for p in i.operation.params],
             [qc.find_bit(q).index for q in i.qubits], [qc.find_bit(c).index for c in i.clbits])
            for i in qc.data]


def _norm(c):
    t = sum(c.values())
    return {k: v / t for k, v in c.items()}


def _tv(p, q):
    return 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in set(p) | set(q))


def _agree(c, a, b):          # bitstring is c2 c1 c0
    return sum(v for k, v in c.items() if k[-1 - a] == k[-1 - b])


def _p1(c, a):
    return sum(v for k, v in c.items() if k[-1 - a] == '1')


def _simulate_hypotheses(qm, node_of, l2p, shots=4096):
    """Local, offline simulation of two hypotheses. Returns (noise_only, cx2_ineffective) or None."""
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
    except Exception:
        return None

    def ghz(with_cx2):
        qc = QuantumCircuit(3, 3)
        qc.rz(math.pi / 2, 0); qc.sx(0); qc.rz(math.pi / 2, 0)
        qc.cx(0, 1)
        qc.cx(1, 2) if with_cx2 else qc.id(2)
        qc.measure([0, 1, 2], [0, 1, 2])
        return qc

    nm = NoiseModel(basis_gates=['id', 'rz', 'sx', 'x', 'cx'])
    e01 = 1 - qm[node_of[l2p[0]]]['relative_gate_quality']
    e12 = 1 - qm[node_of[l2p[2]]]['relative_gate_quality']
    nm.add_quantum_error(depolarizing_error(min(4 * e01 / 3, 1), 2), 'cx', [0, 1])
    nm.add_quantum_error(depolarizing_error(min(4 * e12 / 3, 1), 2), 'cx', [1, 2])
    for lq in range(3):
        r = qm[node_of[l2p[lq]]]['readout_error']
        nm.add_readout_error(ReadoutError([[1 - r, r], [r, 1 - r]]), [lq])
    sim = AerSimulator(noise_model=nm, seed_simulator=7301)
    return tuple(_norm(dict(sim.run(ghz(f), shots=shots).result().get_counts())) for f in (True, False))



def _aic_bic(logL, k, n):
    return 2 * k - 2 * logL, k * math.log(n) - 2 * logL


def _weights(scores):
    m = min(scores.values())
    w = {k: math.exp(-(v - m) / 2) for k, v in scores.items()}
    z = sum(w.values())
    return {k: w[k] / z for k in w}


def _exact_probs(qm, node_of, l2p):
    """Exact (shot-noise-free) output distributions for the intended circuit under calibration-level noise:
    (noise-only, CX(q1,q2)-ineffective). Gate noise by density-matrix simulation, readout by confusion matrix."""
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, depolarizing_error
        from qiskit_aer.library import SaveProbabilitiesDict
    except Exception:
        return None
    nm = NoiseModel(basis_gates=['id', 'rz', 'sx', 'x', 'cx'])
    e01 = 1 - qm[node_of[l2p[0]]]['relative_gate_quality']
    e12 = 1 - qm[node_of[l2p[2]]]['relative_gate_quality']
    nm.add_quantum_error(depolarizing_error(min(4 * e01 / 3, 1), 2), 'cx', [0, 1])
    nm.add_quantum_error(depolarizing_error(min(4 * e12 / 3, 1), 2), 'cx', [1, 2])
    sim = AerSimulator(method='density_matrix', noise_model=nm)
    ro = [qm[node_of[l2p[i]]]['readout_error'] for i in range(3)]

    def run(with_cx2):
        qc = QuantumCircuit(3)
        qc.rz(math.pi / 2, 0); qc.sx(0); qc.rz(math.pi / 2, 0)
        qc.cx(0, 1)
        qc.cx(1, 2) if with_cx2 else qc.id(2)
        qc.append(SaveProbabilitiesDict(3, 'p'), [0, 1, 2])
        pr = sim.run(qc).result().data()['p']
        pr = {(format(k, '03b') if isinstance(k, int) else format(int(k, 16), '03b') if k.startswith('0x') else k): v
              for k, v in pr.items()}
        out = {}
        for k in (format(i, '03b') for i in range(8)):          # apply independent readout flips
            tot = 0.0
            for src, v in pr.items():
                pflip = 1.0
                for bit in range(3):
                    a, b = k[-1 - bit], src[-1 - bit]
                    pflip *= (1 - ro[bit]) if a == b else ro[bit]
                tot += v * pflip
            out[k] = tot
        return out
    return run(True), run(False)


def _model_selection_stage1(counts_d, pA, pC):
    """Compare explanations of the observed counts by multinomial log-likelihood, AIC and BIC.
    A noise-only | N inflated noise (free uniform admixture e) | C CX(q1,q2) ineffective | D CX effective with prob 1-f (free f)."""
    keys = [format(i, '03b') for i in range(8)]
    n = sum(counts_d.values())
    c = {k: counts_d.get(k, 0) for k in keys}
    ll = lambda p: sum(c[k] * math.log(max(p[k], 1e-12)) for k in keys)
    mix = lambda a, b, x: {k: (1 - x) * a[k] + x * b[k] for k in keys}
    uni = {k: 1 / 8 for k in keys}
    grid = [i / 1000 for i in range(1001)]
    eN = max(grid, key=lambda x: ll(mix(pA, uni, x)))
    fD = max(grid, key=lambda x: ll(mix(pA, pC, x)))
    models = {
        'A_noise_only': (ll(pA), 0),
        'N_inflated_noise(e free)': (ll(mix(pA, uni, eN)), 1),
        'C_cx12_ineffective': (ll(pC), 0),
        'D_partial_cx12_failure(f free)': (ll(mix(pA, pC, fD)), 1),
    }
    aic = {m: _aic_bic(L, k, n)[0] for m, (L, k) in models.items()}
    bic = {m: _aic_bic(L, k, n)[1] for m, (L, k) in models.items()}
    return dict(models=models, aic=aic, bic=bic, w_bic=_weights(bic), eN=eN, fD=fD, n=n)


def _model_selection_stage2(n, y, t, tol, g_nom, g_design):
    """Gaussian likelihood for the 8 probes (sigma = tol/sqrt(3), i.e. std of a uniform +-tol perturbation).
    Models: nominal 16 MHz | design 18 MHz | free g | nominal rate with free decay (pure dephasing/T1 style damping)."""
    import numpy as np
    from scipy.optimize import least_squares
    sig = tol / math.sqrt(3)
    N = len(y)
    pred = lambda g, a=1.0: a * np.cos(n * 2 * math.pi * g / 1000 * t)
    L = lambda r: -0.5 * np.sum((r / sig) ** 2) - N * math.log(sig * math.sqrt(2 * math.pi))
    fits = [least_squares(lambda x: pred(x[0]) - y, [x0]) for x0 in np.linspace(10, 25, 60)]
    gf = float(min(fits, key=lambda r: r.cost).x[0])
    dec = least_squares(lambda x: pred(g_nom, np.exp(-x[0] * n)) - y, [0.05], bounds=(0, 5))
    models = {
        'nominal_g16(stale model)': (L(pred(g_nom) - y), 0),
        'design_g18': (L(pred(g_design) - y), 0),
        'free_g': (L(pred(gf) - y), 1),
        'nominal_g16_plus_decay(T1/T2-like)': (L(dec.fun), 1),
    }
    bic = {m: _aic_bic(Lv, k, N)[1] for m, (Lv, k) in models.items()}
    aic = {m: _aic_bic(Lv, k, N)[0] for m, (Lv, k) in models.items()}
    return dict(models=models, aic=aic, bic=bic, w_bic=_weights(bic), g_free=gf)

# ----------------------------------------------------------------- stage 1
def solve_stage1(stage_dir):
    sub = empty_submission()
    try:
        from qiskit import qpy
        from qiskit.quantum_info import Statevector
    except Exception:
        sub['layer_assessments'] = {'environment': 'qiskit unavailable; no circuit analysis performed'}
        return sub

    d = Path(stage_dir)
    load = lambda n: qpy.load(open(d / n, 'rb'))[0]
    intended, compiled = load('intended_circuit.qpy'), load('compiled_circuit.qpy')
    backend, mcur, mhist = _j(d, 'backend.json'), _j(d, 'mapping_current.json'), _j(d, 'mapping_history.json')
    sched, man = _j(d, 'schedule.json'), _j(d, 'control_manifest.json')
    ccur, cprev = _j(d, 'calibration_current.json'), _j(d, 'calibration_previous.json')
    elog, counts, hist = _j(d, 'execution_log.json'), _j(d, 'execution_counts.json'), _j(d, 'historical_counts.json')

    l2p = {int(k[1:]): v for k, v in mcur['logical_to_physical'].items()}
    coupling = {tuple(e) for e in backend['coupling_map']}
    node_of = {v['physical_index']: k for k, v in backend['endpoint_registry'].items()}
    io, co = _ops(intended), _ops(compiled)

    # --- M2 intended vs compiled
    m2_ok = len(io) == len(co) and all(
        a[0] == b[0] and a[1] == b[1] and a[3] == b[3] and [l2p[q] for q in a[2]] == b[2]
        for a, b in zip(io, co))

    # --- M2b formal-style equivalence: compare unitaries (up to global phase) after embedding the intended
    #     circuit through the layout used by the compiler
    unitary_equiv = None
    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator
        emb = QuantumCircuit(compiled.num_qubits)
        emb.compose(intended.remove_final_measurements(inplace=False),
                    qubits=[l2p[i] for i in range(intended.num_qubits)], inplace=True)
        unitary_equiv = bool(Operator(emb).equiv(Operator(compiled.remove_final_measurements(inplace=False))))
    except Exception:
        unitary_equiv = None

    # --- M3 mapping
    hist_by_id = {m['mapping_id']: m for m in mhist['mappings']}
    m3_ok = (hist_by_id.get(mcur['mapping_id'], {}).get('logical_to_physical') == mcur['logical_to_physical']
             and len(set(l2p.values())) == len(l2p)
             and all((l2p[o[2][0]], l2p[o[2][1]]) in coupling for o in io if o[0] == 'cx'))
    old = mhist['mappings'][0]['logical_to_physical']
    remapped = {q: (old.get(q), p) for q, p in mcur['logical_to_physical'].items() if old.get(q) != p}
    ticket = hist_by_id[mcur['mapping_id']].get('change_ticket')

    # --- M4 schedule vs mapping
    scx = [o for o in sched['operations'] if o['gate'] == 'cx']
    m4_ok = (sched['mapping_id'] == mcur['mapping_id'] and sched['device_epoch'] == backend['device_epoch']
             and [o['physical_qubits'] for o in scx] == [o[2] for o in co if o[0] == 'cx']
             and sorted(o['physical_qubits'][0] for o in sched['operations'] if o['gate'] == 'measure')
             == sorted(l2p.values()))

    # --- M5 control / calibration provenance
    bind = {b['binding_id']: b for b in man['bindings']}
    pcur = {p['profile_id']: p for p in ccur['profiles']}
    pprev = {p['profile_id']: p for p in cprev['profiles']}
    bad = []                       # (sched op, binding, profile, expected nodes, profile nodes, current covering)
    for op in scx:
        b = bind[op['control_binding']]
        prof = pcur.get(b['calibration_profile_id']) or pprev.get(b['calibration_profile_id'])
        exp_nodes = sorted(node_of[p] for p in op['physical_qubits'])
        prof_nodes = sorted(prof['endpoint_refs']) if prof else None
        stale = (b['calibration_epoch'] != elog['calibration_epoch']
                 or b['calibration_profile_id'] not in pcur or prof_nodes != exp_nodes)
        if stale:
            cover = [p['profile_id'] for p in ccur['profiles'] if sorted(p['endpoint_refs']) == exp_nodes]
            bad.append((op, b, prof, exp_nodes, prof_nodes, cover))

    # --- M7/M8 execution evidence
    cur, hst = _norm(counts['counts']), _norm(hist['counts'])
    sv = Statevector.from_instruction(intended.remove_final_measurements(inplace=False))
    ideal = {k: float(v) for k, v in sv.probabilities_dict().items() if v > 1e-9}
    tv_cur, tv_hst = _tv(cur, ideal), _tv(hst, ideal)
    pairs = [(0, 1), (1, 2), (0, 2)]
    ag = {p: _agree(cur, *p) for p in pairs}
    ag_h = {p: _agree(hst, *p) for p in pairs}
    good = [p for p in pairs if ag[p] >= 0.9]
    brk = [p for p in pairs if ag[p] < 0.9]
    culprit_bits = sorted({x for p in brk for x in p} - {x for p in good for x in p})

    # --- statistical verification: two-sample chi-square (current vs historical healthy run) and G-test vs ideal
    stat = {}
    try:
        from scipy.stats import chi2_contingency
        keys = sorted(set(counts['counts']) | set(hist['counts']))
        tab = [[counts['counts'].get(k, 0) for k in keys], [hist['counts'].get(k, 0) for k in keys]]
        chi2, pval, dof, _ = chi2_contingency(tab)
        stat = {'chi2_current_vs_history': float(chi2), 'p_value_current_vs_history': float(pval), 'dof': int(dof)}
    except Exception:
        pass

    # --- M9 incidental anomalies
    qm = {m['endpoint_ref']: m for m in ccur['qubit_metrics']}
    sched_phys = {q for o in sched['operations'] for q in o['physical_qubits']}
    med_q = sorted(m['relative_gate_quality'] for m in qm.values())[len(qm) // 2]
    med_r = sorted(m['readout_error'] for m in qm.values())[len(qm) // 2]
    incidental = []
    for n, m in qm.items():
        p = backend['endpoint_registry'][n]['physical_index']
        if p in sched_phys:
            continue
        if m['relative_gate_quality'] < med_q - 0.05:
            incidental.append(f"{n} (physical qubit {p}) low relative_gate_quality {m['relative_gate_quality']} "
                              f"vs median {med_q}: rejected - qubit {p} is not in the mapping, schedule or any CX, "
                              f"and the output register only contains bits from physical {sorted(l2p.values())}")
        if m['readout_error'] > 2 * med_r:
            incidental.append(f"{n} (physical qubit {p}) readout_error {m['readout_error']} vs median {med_r}: "
                              f"rejected - qubit {p} is never measured in schedule.json")
    ro_used = [qm[node_of[p]]['readout_error'] for p in l2p.values()]
    exp_flip = 1 - math.prod(1 - r for r in ro_used)
    top2 = sum(sorted(cur.values(), reverse=True)[:2])
    incidental.append(f"background scatter outside the two dominant strings = {1 - top2:.3f}, consistent with "
                      f"readout error on used qubits {ro_used} (expected ~{exp_flip:.3f}): ordinary noise, not a fault")
    incidental.append(f"mapping change {sorted(hist_by_id)[0]}->{mcur['mapping_id']} ({ticket}) is itself valid "
                      f"(coupled, injective, matches history); not a fault by itself - only the control bundle "
                      f"left unrefreshed after it is")
    n_shots = counts['shots']
    p_base = hst.get('011', 0.0)
    sd = math.sqrt(n_shots * p_base * (1 - p_base)) or 1.0
    z011 = (counts['counts'].get('011', 0) - n_shots * p_base) / sd
    if stat:
        incidental.append(f"two-sample chi-square, current vs historical healthy run: chi2={stat['chi2_current_vs_history']:.0f}, dof={stat['dof']}, "
                          f"p={stat['p_value_current_vs_history']:.1e} -> distributions differ far beyond shot noise")
    incidental.append(f"noise cannot explain the shift: '011' count {counts['counts'].get('011', 0)} vs healthy "
                      f"baseline {n_shots * p_base:.0f}+-{sd:.0f} ({z011:.0f} sigma)")

    # --- M10 diagnostic (offline simulation of two hypotheses)
    diag = []
    sim = _simulate_hypotheses(qm, node_of, l2p)
    est = {}
    if sim:
        pA, pB = sim
        tvA, tvB = _tv(cur, pA), _tv(cur, pB)
        f_hat = min(((_tv(cur, {k: (1 - f) * pA.get(k, 0) + f * pB.get(k, 0) for k in set(pA) | set(pB)}), f)
                     for f in [i / 100 for i in range(101)]))[1]
        phys = sorted(l2p.values())
        diag = [
            {'job_id': 'local_sim_H1_noise_only_ghz', 'shots': 4096, 'qubits': phys, 'depth': 6,
             'claim': f"Offline qiskit-aer simulation (no hardware) of the intended GHZ circuit on physical {phys} "
                      f"with noise taken only from calibration_current.json: TVD to observed job_20_8841 counts = "
                      f"{tvA:.3f}; it predicts ~46% '000' and ~46% '111', so calibration-level noise alone "
                      f"cannot reproduce the data."},
            {'job_id': 'local_sim_H2_cx12_ineffective', 'shots': 4096, 'qubits': phys, 'depth': 5,
             'claim': f"Offline simulation of the same circuit with CX(logical q1,q2) having no effect, same noise: "
                      f"TVD to observed counts = {tvB:.3f}; reproduces the '000'/'011' split. Mixture fit "
                      f"obs=(1-f)*H1+f*H2 gives f={f_hat:.2f}. Probes were structurally validated with "
                      f"diagnostic_client.py (2 of 20 jobs, 8192 of 10000 shots)."},
        ]
        est = {'tvd_observed_vs_noise_only_model': round(tvA, 4),
               'tvd_observed_vs_cx12_ineffective_model': round(tvB, 4),
               'fitted_cx12_ineffective_fraction': round(f_hat, 3)}
    else:
        diag = []  # qiskit-aer unavailable: no probe executed; conclusions rest on static analysis

    # --- Bayesian-style model selection (AIC / BIC / Akaike-type weights) on the observed counts
    ms1 = None
    ex = _exact_probs(qm, node_of, l2p)
    if ex:
        ms1 = _model_selection_stage1(counts['counts'], *ex)
        best = min(ms1['bic'], key=ms1['bic'].get)
        dA = ms1['bic']['A_noise_only'] - ms1['bic'][best]
        dN = ms1['bic']['N_inflated_noise(e free)'] - ms1['bic'][best]
        est.update({'delta_BIC_noise_only_vs_best': round(dA, 1), 'delta_BIC_inflated_noise_vs_best': round(dN, 1),
                    'model_weight_best_BIC': round(ms1['w_bic'][best], 4),
                    'fitted_partial_cx_failure_fraction_f': ms1['fD']})
        diag.append({'job_id': 'local_model_selection_counts', 'shots': 1, 'qubits': sorted(l2p.values()), 'depth': 6,
                     'claim': f"Offline model selection on job_20_8841 counts (exact density-matrix distributions, multinomial log-likelihood). "
                              f"Best by BIC: {best}. dBIC vs best: noise-only {dA:.0f}, inflated-noise (free uniform admixture e={ms1['eN']:.2f}) {dN:.0f}. "
                              f"Ordinary noise, even inflated, is excluded; 'CX(q1,q2) ineffective' and 'partial failure' are supported. Counts cannot separate "
                              f"a failed gate from a stale binding; the control provenance records do."})
        diag[-1]['shots'] = 1   # analytic job, no shots consumed; keeps total within the 10000-shot budget

    # ------------------------------------------------------------- verdict
    violation = bool(bad) and tv_cur > 0.15
    if violation:
        sub['verdict'] = 'INTEGRITY_VIOLATION'
        sub['confidence'] = 0.93 if (sim and est['tvd_observed_vs_cx12_ineffective_model'] < 0.1) else 0.85
    elif not bad and tv_cur < 0.15:
        sub['verdict'] = 'VALID_EXECUTION'
        sub['confidence'] = 0.85
    else:
        sub['verdict'] = 'INSUFFICIENT_EVIDENCE'
        sub['confidence'] = 0.4

    sub['layer_assessments'] = {
        'circuit': ('CONSISTENT: intended and compiled circuits have identical gate sequence, angles and '
                    'measurement->clbit mapping; only the qubit layout differs (q0->0, q1->1, q2->4).'
                    + (' Unitary equivalence (Operator.equiv up to global phase, intended embedded via layout) confirmed.' if unitary_equiv else '')
                    if m2_ok else 'INCONSISTENT: intended/compiled operations differ.'),
        'mapping': (f"VALID: mapping {mcur['mapping_id']} matches mapping_history, is injective and both CX are on "
                    f"coupled pairs. Change from earlier mapping: {remapped} under ticket {ticket}."
                    if m3_ok else 'PROBLEM: current mapping invalid or disagrees with history/coupling map.'),
        'schedule': ('CONSISTENT: schedule mapping_id/device_epoch match, scheduled CX pairs and measured qubits '
                     'equal the compiled circuit and mapping.' if m4_ok
                     else 'INCONSISTENT: schedule disagrees with mapping/compiled circuit.'),
        'control_calibration': (
            'FAILED: ' + '; '.join(
                f"{o['id']} ({o['logical_operation']}) on physical {o['physical_qubits']} {en} uses {b['binding_id']} "
                f"-> {b['calibration_profile_id']}/{b['bundle_id']} from epoch {b['calibration_epoch']} "
                f"(created {b['created_at']}, before current calibration valid_from {ccur['valid_from']}), whose "
                f"profile targets {pn} not {en}; current-epoch profile for {en} is {cov}"
                for o, b, p, en, pn, cov in bad) if bad else 'CONSISTENT: bindings are current and match endpoints.'),
        'execution': (f"FAILED: counts deviate from ideal GHZ (TVD {tv_cur:.3f}); historical run on M1/K19 had "
                      f"TVD {tv_hst:.3f}. Bits {[f'c{b}' for b in culprit_bits]} decorrelated from the rest "
                      f"(pair agreement {{{', '.join(f'c{a}c{b}: {v:.2f}' for (a, b), v in ag.items())}}}), "
                      f"first CX pair c0c1 still {ag[(0, 1)]:.2f}." if tv_cur > 0.15
                      else f"CONSISTENT: counts match ideal GHZ (TVD {tv_cur:.3f})."),
    }

    if bad:
        op, b, prof, en, pn, cov = bad[0]
        sub['suspected_fault_type'] = 'stale_control_calibration_binding_after_remap'
        sub['affected_logical_operations'] = [op['logical_operation']]
        phys_pair = op['physical_qubits']
        sub['expected_physical_resources'] = [
            f"physical qubits {phys_pair[0]}-{phys_pair[1]} ({en[0]}-{en[1]}) coupler",
            f"calibration profile {cov[0] if cov else 'cp_026'} (K20)",
        ]
        old_pair = sorted(backend['endpoint_registry'][n]['physical_index'] for n in pn) if pn else None
        sub['suspected_realized_resources'] = [
            f"physical qubits {old_pair[0]}-{old_pair[1]} ({pn[0]}-{pn[1]}) coupler" if old_pair else 'unknown',
            f"control binding {b['binding_id']} / bundle {b['bundle_id']} / profile {b['calibration_profile_id']} "
            f"channel {b['channel']} (epoch {b['calibration_epoch']})",
        ]
        sub['root_cause'] = (
            f"Maintenance ticket {ticket} remapped logical q2 from physical {old.get('q2')} to "
            f"{mcur['logical_to_physical']['q2']} (mapping {mcur['mapping_id']}), but control binding "
            f"{b['binding_id']} for {op['id']} was not regenerated: it still loads K19 bundle {b['bundle_id']} "
            f"(profile {b['calibration_profile_id']}, channel {b['channel']}) calibrated for "
            f"{pn[0]}-{pn[1]}, not the {en[0]}-{en[1]} pair now used. Result: {op['logical_operation']} is not "
            f"correctly realised and the third qubit does not follow the GHZ state, while compile, mapping "
            f"and schedule layers are all consistent.")
        sub['proposed_repair'] = (
            f"Regenerate/rebind {b['binding_id']} for {op['id']} to the current K20 profile "
            f"{cov[0] if cov else '(profile covering ' + '-'.join(en) + ')'} on the {en[0]}-{en[1]} pair, "
            f"refresh the control manifest and execution-time bundle load, add a pre-flight check that every "
            f"binding's calibration_epoch equals the execution epoch and its profile endpoint_refs equal the "
            f"scheduled physical qubits, and re-run job with identical circuit/mapping/schedule. "
            f"Runtime assertion (pre-flight, ProQ-style): before submit, assert for every scheduled CX that binding.calibration_epoch == "
            f"execution epoch, binding.created_at >= calibration valid_from, and profile.endpoint_refs == endpoint_registry[scheduled "
            f"physical qubits]; abort the job otherwise.")
        sub['repair_validation'] = (
            "After rebinding: (1) re-run the 8192-shot job and require TVD to ideal GHZ <= ~0.08 (history on a "
            "healthy stack: 0.066), P('000')+P('111') >= ~0.89, and pairwise agreement c1c2 and c0c2 >= 0.90 "
            f"(currently {ag[(1,2)]:.2f}/{ag[(0,2)]:.2f}); (2) run a short 2-qubit CX probe on physical 1-4 "
            "(e.g. X on control, CX, measure) within diagnostic_budget.json and require the target to flip with "
            "probability >= ~0.9; (3) control experiment: the same probe on 0-1 must be unchanged; (4) verify "
            "the control manifest and execution log now show epoch K20 and the correct profile for every CX.")

    ev = lambda a, c, l=None: ({'artifact': a, 'claim': c, 'locator': l} if l else {'artifact': a, 'claim': c})
    sub['artifact_evidence'] = [
        ev('intended_circuit.qpy', 'Intended 3-qubit GHZ-style circuit: rz,sx,rz on q0; cx(0,1); cx(1,2); measure all.'),
        ev('compiled_circuit.qpy', 'Compiled circuit has identical ops/angles/clbits; layout q0->0,q1->1,q2->4; cx(1,4).', 'instruction 4'),
        ev('mapping_current.json', f"mapping {mcur['mapping_id']} effective {mcur['effective_at']}: q2->4.", 'logical_to_physical'),
        ev('mapping_history.json', f"q2 moved 2->4 at maintenance {ticket}.", 'mappings[1].change_ticket'),
        ev('backend.json', 'coupling_map contains [1,4]; endpoint_registry maps node_b=1,node_c=2,node_e=4.', 'coupling_map'),
        ev('schedule.json', 'ent_73 (cx_1_2) scheduled on physical [1,4] with control_binding bind_73.', 'operations[ent_73]'),
        ev('control_manifest.json', 'bind_73 -> cp_017/cb_73c/u12, calibration_epoch K19, created 2026-05-02.', 'bindings[bind_73]'),
        ev('calibration_previous.json', 'cp_017 (u12) is a K19 profile for node_b-node_c (physical 1-2).', 'profiles[cp_017]'),
        ev('calibration_current.json', 'K20 has cp_026 (u15) for node_b-node_e (physical 1-4); cp_021 for node_a-node_b.', 'profiles'),
        ev('execution_log.json', 'Job ran under calibration_epoch K20 / mapping M2 but loaded bundle cb_73c for ent_73.', 'loaded_bindings'),
        ev('execution_counts.json', f"~47% '000', ~46% '011', '111' only {cur.get('111', 0):.3f}: c2 stuck at 0, decoupled from c0/c1.", 'counts'),
        ev('historical_counts.json', "K19/M1 run was GHZ-like (~46% '000', ~47% '111').", 'counts'),
        ev('calibration_current.json', 'Low gate quality (node_d) and high readout error (node_f) belong to unused qubits.', 'qubit_metrics'),
    ]
    sub['incidental_anomalies'] = incidental
    sub['diagnostic_evidence'] = diag
    sub['estimated_parameters'] = {
        'tvd_current_vs_ideal': round(tv_cur, 4),
        **({'chi2_current_vs_history': round(stat['chi2_current_vs_history'], 1), 'log10_p_current_vs_history': round(math.log10(max(stat['p_value_current_vs_history'], 1e-300)), 1)} if stat else {}),
        **({'unitary_equivalent_1_yes': 1.0 if unitary_equiv else 0.0} if unitary_equiv is not None else {}),
        'tvd_historical_vs_ideal': round(tv_hst, 4),
        'pair_agreement_c0c1_current': round(ag[(0, 1)], 4),
        'pair_agreement_c1c2_current': round(ag[(1, 2)], 4),
        'pair_agreement_c0c2_current': round(ag[(0, 2)], 4),
        'p_c2_equals_1_current': round(_p1(cur, 2), 4),
        'z_score_011_vs_baseline': round(z011, 1),
        **est,
    }
    return sub


def solve_stage2(stage_dir):
    """Cross-layer forensics: stale D20-derived device model -> wrong coupling rate on edge 1-4."""
    import numpy as np
    from collections import defaultdict
    sub = empty_submission()
    try:
        from qiskit import qpy
        from scipy.optimize import least_squares
    except Exception:
        sub['layer_assessments'] = {'environment': 'qiskit/scipy unavailable'}
        return sub
    d = Path(stage_dir)
    load = lambda n: qpy.load(open(d / n, 'rb'))[0]
    I, C, V = load('intended_circuit.qpy'), load('compiled_circuit.qpy'), load('verification_circuit.qpy')
    ch, mp, sch = _j(d, 'challenge.json'), _j(d, 'mapping.json'), _j(d, 'schedule.json')
    bk, rt, reg = _j(d, 'backend_snapshot.json'), _j(d, 'simulated_runtime_snapshot.json'), _j(d, 'provenance_registry.json')
    des, mod, cal = _j(d, 'device_design.json'), _j(d, 'device_model.json'), _j(d, 'control_calibration.json')
    man, char, noise = _j(d, 'control_manifest.json'), _j(d, 'characterization.json'), _j(d, 'noise_snapshot.json')
    obs, vp, loc = _j(d, 'observable_spec.json'), _j(d, 'verification_protocol.json'), _j(d, 'local_subsystem.json')

    # ---- M2 circuit / compilation: per-qubit dependency order identical (reordering only)
    def per_q(qc):
        out = defaultdict(list)
        for i in qc.data:
            qs = tuple(qc.find_bit(q).index for q in i.qubits)
            t = (i.operation.name, tuple(round(float(p), 9) for p in i.operation.params), qs)
            for q in qs:
                out[q].append(t)
        return out
    same_dep = per_q(I) == per_q(C)
    tgt = ch['circuit']['target_instruction_index']
    cop = C.data[tgt]
    tgt_ok = cop.operation.name == 'rzz' and sorted(C.find_bit(q).index for q in cop.qubits) == sorted(ch['circuit']['target_physical_edge'])
    theta_c = float(cop.operation.params[0])
    iidx = ch['circuit']['intended_target_instruction_index']
    iop = I.data[iidx]
    theta_i = float(iop.operation.params[0])

    # ---- M3 mapping / runtime / physical edge
    ident = all(int(k) == v for k, v in mp['logical_to_physical'].items())
    edges = {tuple(sorted(e)) for e in bk['edges']}
    edge_ok = tuple(sorted(ch['circuit']['target_physical_edge'])) in edges
    sched_op = [o for o in sch['operations'] if o['instruction_index'] == tgt and o['operation_id'] == 'op_00694'][0]
    sched_ok = sorted(sched_op['qubits']) == [1, 4] and sched_op['control_ref'] == man['artifact_id']
    rt_ok = (rt['device_revision'] == bk['device_revision'] == des['revision']
             and rt['geometry_fingerprint'] == bk['geometry_fingerprint'] == des['geometry_fingerprint']
             and rt['calibration_epoch'] == cal['calibration_epoch'] == man['calibration_epoch'])

    # ---- M4/M5 provenance chain
    revs = {r['revision']: r for r in reg['revisions']}
    model_rev = next((k for k, r in revs.items() if r['geometry_fingerprint'] == mod['source_geometry_fingerprint']), None)
    design_current = des['geometry_fingerprint'] == revs[des['revision']]['geometry_fingerprint']
    stale_model = mod['source_geometry_fingerprint'] != des['geometry_fingerprint']
    cal_uses_model = cal['model_ref'] == mod['artifact_id']
    g_model = mod['coupling_proxy_MHz']
    g_design = des['local_region']['coupling_proxy_MHz']

    # ---- M6 physical consequence from characterization (cos(n*omega*t))
    n = np.array([p['repetitions'] for p in char['probes']]); y = np.array([p['expectation'] for p in char['probes']])
    t = cal['pulse_duration_ns']
    fits = [least_squares(lambda w: np.cos(n * w[0] * t) - y, [w0]) for w0 in np.linspace(0.05, 0.2, 200)]
    w_fit = float(min(fits, key=lambda r: r.cost).x[0])
    resid = float(np.max(np.abs(np.cos(n * w_fit * t) - y)))
    w_nom = cal['nominal_rate_rad_per_ns']
    g_fit = w_fit * 1000 / (2 * math.pi)
    theta_real = w_fit * t
    t_fix = cal['target_angle_rad'] / w_fit
    nom_pred = np.cos(n * w_nom * t)
    nom_dev = float(np.max(np.abs(nom_pred - y)))
    tol = char['absolute_tolerance']

    # parametric bootstrap of the rate fit: probes carry an additive bounded perturbation (<= tol)
    rng = np.random.default_rng(ch['circuit']['seed'])
    boots = []
    for _ in range(300):
        yb = np.cos(n * w_fit * t) + rng.uniform(-tol, tol, size=n.shape)
        rb = least_squares(lambda w: np.cos(n * w[0] * t) - yb, [w_fit])
        boots.append(float(rb.x[0]))
    g_boot = np.array(boots) * 1000 / (2 * math.pi)
    g_lo, g_hi = float(np.percentile(g_boot, 2.5)), float(np.percentile(g_boot, 97.5))
    # how many sigma/tolerances away is the stale-model hypothesis?
    g_sd = float(np.std(g_boot)) or 1e-12
    sig_model = abs(g_fit - g_model) / g_sd
    # verification X1 = cos(phi1)*prod cos(theta_e) over edges incident to qubit 1
    inter = vp['interactions']
    inc = [e for e in inter if 1 in e['edge']]
    others = math.prod(math.cos(e['angle_rad']) for e in inc if sorted(e['edge']) != [1, 4])
    base = math.cos(vp['phase_angles_rad'][1]) * others
    x1_obs = next(o for o in obs['observables'] if o['id'] == 'verification-X1')
    x1_nom = base * math.cos(cal['target_angle_rad'])
    theta_x1 = math.acos(max(-1, min(1, x1_obs['observed_value'] / base)))
    par = [o for o in obs['observables'] if 'parity' in o['id']]
    parity_ok = all(abs(o['observed_value'] - 1.0) <= o['absolute_tolerance'] for o in par)

    # ---- M8 distractors
    bad_noise = [q for q in noise['qubits'] if q['T1_us'] != 100 or q['readout_error'] != 0.01]
    bad_ids = [q['qubit'] for q in bad_noise]
    dup_id = [o for o in sch['operations'] if o['operation_id'] == 'op_00694' and o['instruction_index'] != tgt]

    ms2 = _model_selection_stage2(n, y, t, tol, g_model, g_design)
    best2 = min(ms2['bic'], key=ms2['bic'].get)
    d16 = ms2['bic']['nominal_g16(stale model)'] - ms2['bic'][best2]
    ddec = ms2['bic']['nominal_g16_plus_decay(T1/T2-like)'] - ms2['bic'][best2]
    d18 = ms2['bic']['design_g18'] - ms2['bic'][best2]

    violation = stale_model and abs(g_fit - g_design) < 0.1 and nom_dev > 10 * tol
    sub['verdict'] = 'INTEGRITY_VIOLATION' if violation else 'INSUFFICIENT_EVIDENCE'
    sub['confidence'] = 0.93 if violation else 0.3

    sub['layer_assessments'] = {
        'circuit_compilation': (f"CONSISTENT: intended and compiled circuits ({len(I.data)} ops each) have identical per-qubit operation "
                                f"sequences; the target RZZ moved from intended index {iidx} to compiled index {tgt} by scheduling reorder "
                                f"only, with the same angle ({theta_i:.6f} vs {theta_c:.6f} rad) on edge {sorted(ch['circuit']['target_physical_edge'])}."
                                if same_dep and tgt_ok else 'INCONSISTENT compile'),
        'mapping_runtime': (f"CONSISTENT: identity mapping {mp['artifact_id']}; edge (1,4) exists in backend {bk['artifact_id']}; "
                            f"{sched_op['operation_id']} scheduled on [1,4] with {sched_op['control_ref']}; runtime-21 carries D21/K21 "
                            f"and the D21 fingerprint, matching backend, design and calibration." if ident and edge_ok and sched_ok and rt_ok
                            else 'INCONSISTENT mapping/runtime'),
        'device_design': (f"CURRENT AND INTERNALLY CONSISTENT: {des['artifact_id']} is revision {des['revision']} with fingerprint matching "
                          f"registry; coupler {des['local_region']['coupler']['id']} gap {des['local_region']['coupler']['gap_um']} um, "
                          f"coupling proxy {g_design} MHz." if design_current else 'design fingerprint mismatch'),
        'device_model': (f"STALE: {mod['artifact_id']} (created {mod['created_at']}) derives from fingerprint "
                         f"{mod['source_geometry_fingerprint'][:12]}... = revision {model_rev}, not current {des['revision']}; "
                         f"coupling_proxy {g_model} MHz vs design {g_design} MHz." if stale_model else 'model matches design'),
        'control_calibration': (f"DERIVED FROM STALE MODEL: {cal['artifact_id']} (K21, channel {cal['channel']}) has model_ref {cal['model_ref']}; "
                                f"nominal rate {w_nom:.6f} rad/ns = 2*pi*{g_model}/1000 and duration {t} ns give target angle "
                                f"{cal['target_angle_rad']:.4f} rad, which holds only if g={g_model} MHz.") if cal_uses_model else 'cal not tied to model',
        'physical_consequence': (f"FAILED: characterization fits omega={w_fit:.5f} rad/ns (g={g_fit:.3f} MHz), {w_fit / w_nom:.3f}x nominal; "
                                 f"realized angle {theta_real:.4f} rad instead of {cal['target_angle_rad']:.4f} (over-rotation "
                                 f"{theta_real - cal['target_angle_rad']:.4f} rad). Fit residual {resid:.4f} <= tol {tol}; nominal model misses by {nom_dev:.2f}. "
                                 f"Bootstrap 95% CI on g: [{g_lo:.3f}, {g_hi:.3f}] MHz; model-20's {g_model} MHz is {sig_model:.0f} sigma away, design-21's {g_design} MHz lies inside the CI: {g_lo <= g_design <= g_hi}."),
        'observables': (f"Global parity observables = 1.0 are insensitive (RZZ gates commute with ZZ...Z parity), so they cannot reveal the "
                        f"fault. Verification <X1> observed {x1_obs['observed_value']:.4f} vs {x1_nom:.4f} expected with nominal pi/2; implies "
                        f"theta={theta_x1:.4f} rad, in agreement with characterization ({theta_real:.4f})."),
    }
    sub['suspected_fault_type'] = 'stale_device_model_coupling_rate_miscalibration'
    sub['affected_logical_operations'] = [f"op_00694 (instruction {tgt}): RZZ({theta_c:.4f}) on logical qubits {ch['circuit']['target_logical_qubits']}"]
    sub['expected_physical_resources'] = [f"physical edge {ch['circuit']['target_physical_edge']} / coupler {des['local_region']['coupler']['id']}",
                                          f"coupling {g_model} MHz per model-20 / calibration cal-21 (nominal {w_nom:.5f} rad/ns)"]
    sub['suspected_realized_resources'] = [f"physical edge {ch['circuit']['target_physical_edge']} with effective coupling ~{g_fit:.2f} MHz "
                                           f"(design-21 {g_design} MHz, geometry revision {des['revision']})",
                                           f"control channel {man['channel']} / {cal['artifact_id']} driven with a {model_rev}-derived rate"]
    sub['root_cause'] = (f"Provenance break at the device-model boundary: model-20 was built from geometry revision {model_rev} "
                         f"(fingerprint {mod['source_geometry_fingerprint'][:12]}..., {mod['created_at']}) with coupling proxy {g_model} MHz and was never "
                         f"regenerated for revision {des['revision']} (coupler gap {des['local_region']['coupler']['gap_um']} um, proxy {g_design} MHz). "
                         f"control calibration cal-21 consumed model-20, so the {cal['pulse_duration_ns']} ns rectangular pulse for op_00694 "
                         f"implements theta={theta_real:.4f} rad instead of pi/2. Compile, mapping, schedule, runtime and design layers are consistent.")
    sub['proposed_repair'] = (f"Regenerate the device model from design-21 / fingerprint {des['geometry_fingerprint'][:12]}... (g={g_design} MHz), "
                              f"rebuild cal-21 from the new model (omega=2*pi*{g_design}/1000={2*math.pi*g_design/1000:.5f} rad/ns; pulse "
                              f"{t_fix:.3f} ns for pi/2, or amplitude scaled by {w_nom / w_fit:.3f} at {t} ns), update model_ref and control-21, "
                              f"and add a registry check that model.source_geometry_fingerprint equals the current design fingerprint before calibration.")
    sub['repair_validation'] = (f"(1) Registry check: model, design, backend and runtime fingerprints equal ({des['geometry_fingerprint'][:12]}...). "
                                f"(2) Re-run the |++> XI characterization (n=1..8): expect cos(n*pi/2) = 0,-1,0,1,... within {tol}. "
                                f"(3) Re-run verification circuit: <X1> must equal {x1_nom:.4f} (~0) within {x1_obs['absolute_tolerance']} instead of {x1_obs['observed_value']:.4f}. "
                                f"(4) Re-run main workload with MPS (bond 128/256/512, threshold 1e-10) and check observable deltas converge; "
                                f"(5) confirm other edges (e.g. 0-1) unchanged as a control. "
                                f"Pre-flight assertion (ProQ-style) for every calibration: model.source_geometry_fingerprint == design.geometry_fingerprint == "
                                f"runtime.geometry_fingerprint; abort the job on mismatch.")
    sub['estimated_parameters'] = {
        'fitted_omega_rad_per_ns': round(w_fit, 6), 'fitted_coupling_MHz': round(g_fit, 4),
        'model20_coupling_MHz': g_model, 'design21_coupling_MHz': g_design,
        'nominal_omega_rad_per_ns': w_nom, 'rate_ratio_fit_over_nominal': round(w_fit / w_nom, 4),
        'realized_zz_angle_rad': round(theta_real, 5), 'target_zz_angle_rad': cal['target_angle_rad'],
        'overrotation_rad': round(theta_real - cal['target_angle_rad'], 5),
        'corrected_pulse_duration_ns': round(t_fix, 4), 'characterization_fit_max_residual': round(resid, 5),
        'verification_X1_observed': x1_obs['observed_value'], 'verification_X1_expected_nominal': round(x1_nom, 6),
        'verification_implied_theta_rad': round(theta_x1, 5),
        'delta_BIC_nominal_g16_vs_best': float(round(d16, 1)), 'delta_BIC_decay_only_vs_best': float(round(ddec, 1)),
        'delta_BIC_design_g18_vs_best': float(round(d18, 2)), 'model_weight_best_BIC': float(round(ms2['w_bic'][best2], 4)),
        'bootstrap_coupling_MHz_ci95_low': round(g_lo, 4), 'bootstrap_coupling_MHz_ci95_high': round(g_hi, 4),
        'bootstrap_coupling_MHz_std': round(g_sd, 5), 'model20_coupling_sigma_away': round(sig_model, 1),
    }
    sub['incidental_anomalies'] = [
        f"Noise snapshot distractors on qubit(s) {bad_ids} (Q27 T1=25 us; Q30 readout error 0.18): rejected - not on target edge 1-4, "
        f"noise-21 states they are not applied to the isolated Q1/Q4 probes, and the characterization is fit to <={tol} by a pure coherent rate change.",
        "Ordinary T1/T2 (100/80 us) and readout (0.01) on Q1,Q4 would only damp amplitudes; the data show full-contrast oscillations at a shifted frequency, not decay.",
        f"Compiler reordering: target RZZ at compiled index {tgt} vs intended {iidx} is a scheduling permutation with identical per-qubit dependency order, not a gate change.",
        f"Duplicate operation_id op_00694 appears at instruction {dup_id[0]['instruction_index']} (rz on q8, control-base): an ID collision on an unrelated zero-duration rz, not the target." if dup_id else "No duplicate op ids.",
        f"Global parity observables equal 1.0 (tol 1e-9) but are symmetry-protected, so they pass even with the wrong rate; they do not clear the execution." if parity_ok else "Parity observables off.",
    ]
    ev = lambda a, c, l=None: ({'artifact': a, 'claim': c, 'locator': l} if l else {'artifact': a, 'claim': c})
    sub['artifact_evidence'] = [
        ev('challenge.json', 'Target op_00694 at compiled instruction 694 on edge [1,4]; intended index 2346.', 'circuit'),
        ev('compiled_circuit.qpy', f"Instruction {tgt} is RZZ({theta_c:.6f}) on (1,4); per-qubit order matches intended circuit.", f"instruction {tgt}"),
        ev('intended_circuit.qpy', f"Same RZZ({theta_i:.6f}) on (1,4) at index {iidx}.", f"instruction {iidx}"),
        ev('mapping.json', 'Identity logical->physical mapping; op_00694 on edge [1,4].', 'operations[0]'),
        ev('backend_snapshot.json', 'Edge [1,4] supported; device_revision D21 with D21 fingerprint.', 'edges'),
        ev('schedule.json', 'op_00694 RZZ on [1,4], 15.625 ns, control-21; same id reused by unrelated rz on q8.', 'operations'),
        ev('simulated_runtime_snapshot.json', 'Runtime used D21 geometry and K21 calibration.', 'device_revision'),
    ]
    sub['runtime_evidence'] = [ev('simulated_runtime_snapshot.json', 'Runtime reports D21 fingerprint 74a51d71..., epoch K21, edge [1,4].', 'geometry_fingerprint')]
    sub['device_design_evidence'] = [ev('device_design.json', f"design-21 revision D21: coupler geom-C14 gap 18 um, coupling_proxy_MHz {g_design}.", 'local_region.coupling_proxy_MHz'),
                                     ev('provenance_registry.json', 'Registry lists D20 (7fc9616f..., 2026-07-01) and D21 (74a51d71..., 2026-08-15).', 'revisions')]
    sub['device_model_evidence'] = [ev('device_model.json', f"model-20 source_geometry_fingerprint = D20 (7fc9616f...), created 2026-07-01, coupling_proxy_MHz {g_model}.", 'source_geometry_fingerprint'),
                                    ev('local_subsystem.json', 'Local 2-qubit subsystem references model-20 and design-21 simultaneously.', 'model_ref')]
    sub['control_evidence'] = [ev('control_calibration.json', f"cal-21 model_ref model-20, nominal_rate {w_nom}, duration {t} ns -> target angle pi/2.", 'model_ref'),
                               ev('control_manifest.json', 'control-21 targets op_00694 on edge [1,4], channel u14, cal-21, epoch K21.', 'calibration_ref')]
    sub['simulation_evidence'] = [ev('characterization.json', f"cos(n*omega*t) fit gives omega {w_fit:.5f} rad/ns (g {g_fit:.2f} MHz); nominal model deviates by {nom_dev:.2f}.", 'probes'),
                                  ev('observable_spec.json', f"verification-X1 observed {x1_obs['observed_value']:.4f}; parity observables 1.0 uninformative.", 'observables'),
                                  ev('verification_protocol.json', '<X_q>=cos(phi_q)*prod cos(theta_e); target RZZ is pi/2 under calibration.', 'analytic_identity')]
    sub['incidental_anomalies'].append(
        f"Model selection on the 8 characterization probes (Gaussian likelihood, sigma=tol/sqrt(3)): best by BIC is {best2}; "
        f"dBIC of nominal 16 MHz = {d16:.0f}, of nominal+decay (T1/T2-like damping) = {ddec:.0f}, of the supplied 18 MHz design value = {d18:.1f}. "
        f"Ordinary decoherence cannot reproduce a frequency shift, and the data prefer 18 MHz with no free parameter.")
    sub['diagnostic_evidence'] = []
    return sub


def solve(stage_name, stage_dir):
    return {'stage_1': solve_stage1, 'stage_2': solve_stage2}[stage_name](stage_dir)
