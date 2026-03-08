import datetime
import argparse
import numpy as np
import scipy.sparse as sp
import pyscipopt as scip
import pickle
import gzip
import ecole

# ---------- Ecole observation functions ----------
_NODE_BIPARTITE_OBS = ecole.observation.NodeBipartite(cache=False)
_SB_SCORES_OBS = ecole.observation.StrongBranchingScores(pseudo_candidates=False)

def _as_ecole_model(model):
    """
    Convert PySCIPOpt model to Ecole model sharing the same SCIP state.
    """
    return ecole.scip.Model.from_pyscipopt(model)

def _safe_var_name(v):
    try:
        return v.name
    except Exception:
        try:
            return v.getName()
        except Exception:
            return None

def _var_probindex(v):
    """
    Best-effort variable problem index for aligning with Ecole scores.
    """
    for name in ["getProbindex", "getProbIndex", "getIndex"]:
        if hasattr(v, name):
            try:
                idx = int(getattr(v, name)())
                if idx >= 0:
                    return idx
            except Exception:
                pass

    # fallback through column lp position if needed
    try:
        col = v.getCol()
        if hasattr(col, "getLPPos"):
            idx = int(col.getLPPos())
            if idx >= 0:
                return idx
    except Exception:
        pass

    return None

def log(str, logfile=None):
    str = f'[{datetime.datetime.now()}] {str}'
    print(str)
    if logfile is not None:
        with open(logfile, mode='a') as f:
            print(str, file=f)


def init_scip_params(model, seed, heuristics=True, presolving=True, separating=True, conflict=True):

    seed = seed % 2147483648  # SCIP seed range

    # set up randomization
    model.setBoolParam('randomization/permutevars', True)
    model.setIntParam('randomization/permutationseed', seed)
    model.setIntParam('randomization/randomseedshift', seed)

    # separation only at root node
    model.setIntParam('separating/maxrounds', 0)

    # no restart
    model.setIntParam('presolving/maxrestarts', 0)

    # if asked, disable presolving
    if not presolving:
        model.setIntParam('presolving/maxrounds', 0)
        model.setIntParam('presolving/maxrestarts', 0)

    # if asked, disable separating (cuts)
    if not separating:
        model.setIntParam('separating/maxroundsroot', 0)

    # if asked, disable conflict analysis (more cuts)
    if not conflict:
        model.setBoolParam('conflict/enable', False)

    # if asked, disable primal heuristics
    if not heuristics:
        model.setHeuristics(scip.SCIP_PARAMSETTING.OFF)

def extract_state(model, buffer=None):
    """
    Manual fallback extractor for newer PySCIPOpt versions without getState().
    Returns:
        constraint_features, edge_features, variable_features
    in the same outer format expected by the rest of the pipeline.
    """
    import numpy as np

    def _safe_name(v):
        try:
            return v.name
        except Exception:
            return v.getName()

    def _safe_obj(v):
        try:
            return float(v.getObj())
        except Exception:
            try:
                return float(v.obj)
            except Exception:
                return 0.0

    def _safe_vtype(v):
        try:
            return v.vtype()
        except Exception:
            try:
                return v.getType()
            except Exception:
                return "CONTINUOUS"

    def _safe_lb(v):
        try:
            return v.getLbGlobal()
        except Exception:
            try:
                return v.getLbLocal()
            except Exception:
                return None

    def _safe_ub(v):
        try:
            return v.getUbGlobal()
        except Exception:
            try:
                return v.getUbLocal()
            except Exception:
                return None

    def _safe_sol(model, v):
        try:
            return float(model.getSolVal(None, v))
        except Exception:
            try:
                return float(v.getLPSol())
            except Exception:
                return 0.0

    def _safe_var_name(v):
        try:
            return v.name
        except Exception:
            try:
                return v.getName()
            except Exception:
                return None

    def _safe_col_var_name(col):
        try:
            v = col.getVar()
            return _safe_var_name(v)
        except Exception:
            pass
        try:
            return col.name
        except Exception:
            pass
        try:
            return col.getName()
        except Exception:
            return None

    vars_ = list(model.getVars(transformed=True))
    col_index = {_safe_name(v): i for i, v in enumerate(vars_)}

    obj = np.array([_safe_obj(v) for v in vars_], dtype=np.float32)
    obj_norm = np.linalg.norm(obj)
    if obj_norm <= 0:
        obj_norm = 1.0

    # ---------- variable features ----------
    v_feats = []
    for v in vars_:
        vt = [0.0, 0.0, 0.0, 0.0]  # BINARY, INTEGER, IMPLINT, CONTINUOUS
        vtype = _safe_vtype(v)

        if vtype == "BINARY":
            vt[0] = 1.0
        elif vtype == "INTEGER":
            vt[1] = 1.0
        elif vtype == "IMPLINT":
            vt[2] = 1.0
        else:
            vt[3] = 1.0

        lb = _safe_lb(v)
        ub = _safe_ub(v)

        has_lb = 0.0 if lb is None or np.isneginf(lb) else 1.0
        has_ub = 0.0 if ub is None or np.isposinf(ub) else 1.0

        sol = _safe_sol(model, v)
        sol_at_lb = float(has_lb and abs(sol - lb) <= 1e-6)
        sol_at_ub = float(has_ub and abs(sol - ub) <= 1e-6)

        frac = 0.0
        if vt[3] == 0.0:
            frac = abs(sol - np.round(sol))

        try:
            redcost = float(model.getVarRedcost(v)) / obj_norm
        except Exception:
            redcost = 0.0

        inc_val = 0.0
        avg_inc_val = 0.0
        age = 0.0
        basis = [0.0, 0.0, 0.0, 0.0]

        row = vt + [
            _safe_obj(v) / obj_norm,
            has_lb,
            has_ub,
            sol_at_lb,
            sol_at_ub,
            frac,
            redcost,
            sol,
            inc_val,
            avg_inc_val,
        ] + basis + [age]

        v_feats.append(row)

    v_feats = np.asarray(v_feats, dtype=np.float32)

    variable_features = {
        "names": [f"var_feat_{i}" for i in range(v_feats.shape[1])],
        "values": v_feats,
    }

    # ---------- constraint features + edge features from LP rows ----------
    rows = model.getLPRowsData()

    c_feats = []
    edge_rows = []
    edge_cols = []
    edge_vals = []

    for r_idx, row in enumerate(rows):
        try:
            row_cols = row.getCols()
        except Exception:
            row_cols = []

        try:
            row_vals = row.getVals()
        except Exception:
            row_vals = []

        coeffs = np.array(row_vals, dtype=np.float32) if len(row_vals) else np.zeros(0, dtype=np.float32)
        row_norm = np.linalg.norm(coeffs)
        if row_norm <= 0:
            row_norm = 1.0

        try:
            lhs = row.getLhs()
        except Exception:
            lhs = np.nan

        try:
            rhs = row.getRhs()
        except Exception:
            rhs = np.nan

        try:
            dual = row.getDualsol() / (row_norm * obj_norm)
        except Exception:
            try:
                dual = model.getDualsolLinear(row) / (row_norm * obj_norm)
            except Exception:
                dual = 0.0

        try:
            activity = row.getLPActivity()
        except Exception:
            try:
                activity = model.getRowLPActivity(row)
            except Exception:
                activity = 0.0

        slack = 0.0
        try:
            if np.isfinite(lhs) and np.isfinite(rhs) and abs(lhs - rhs) <= 1e-9:
                slack = abs(activity - rhs)
            elif np.isfinite(rhs):
                slack = rhs - activity
            elif np.isfinite(lhs):
                slack = activity - lhs
        except Exception:
            slack = 0.0

        try:
            obj_par = model.getRowObjParallelism(row)
        except Exception:
            obj_par = 0.0

        is_tight = float(abs(slack) <= 1e-6)

        c_feats.append([
            obj_par,
            activity / row_norm,
            slack / row_norm,
            dual,
            is_tight,
        ])

        for col, coef in zip(row_cols, row_vals):
            var_name = _safe_col_var_name(col)
            if var_name is None:
                continue
            if var_name not in col_index:
                continue

            edge_rows.append(r_idx)
            edge_cols.append(col_index[var_name])
            edge_vals.append([float(coef) / row_norm])

    c_feats = np.asarray(c_feats, dtype=np.float32)

    if len(edge_rows) > 0:
        e_idx = np.vstack([
            np.asarray(edge_rows, dtype=np.int64),
            np.asarray(edge_cols, dtype=np.int64),
        ])
        e_vals = np.asarray(edge_vals, dtype=np.float32)
    else:
        e_idx = np.zeros((2, 0), dtype=np.int64)
        e_vals = np.zeros((0, 1), dtype=np.float32)

    constraint_features = {
        "names": [
            "obj_parallelism",
            "activity_norm",
            "slack_norm",
            "dual_norm",
            "is_tight",
        ],
        "values": c_feats,
    }

    edge_features = {
        "names": ["coef_normalized"],
        "indices": e_idx,
        "values": e_vals,
    }

    return constraint_features, edge_features, variable_features
    
    
def valid_seed(seed):
    """Check whether seed is a valid random seed or not."""
    seed = int(seed)
    if seed < 0 or seed > 2**32 - 1:
        raise argparse.ArgumentTypeError(
                "seed must be any integer between 0 and 2**32 - 1 inclusive")
    return seed


def compute_extended_variable_features(state, candidates):
    """
    Utility to extract variable features only from a bipartite state representation.

    Parameters
    ----------
    state : dict
        A bipartite state representation.
    candidates: list of ints
        List of candidate variables for which to compute features (given as indexes).

    Returns
    -------
    variable_states : np.array
        The resulting variable states.
    """
    constraint_features, edge_features, variable_features = state
    constraint_features = constraint_features['values']
    edge_indices = edge_features['indices']
    edge_features = edge_features['values']
    variable_features = variable_features['values']

    cand_states = np.zeros((
        len(candidates),
        variable_features.shape[1] + 3*(edge_features.shape[1] + constraint_features.shape[1]),
    ))

    # re-order edges according to variable index
    edge_ordering = edge_indices[1].argsort()
    edge_indices = edge_indices[:, edge_ordering]
    edge_features = edge_features[edge_ordering]

    # gather (ordered) neighbourhood features
    nbr_feats = np.concatenate([
        edge_features,
        constraint_features[edge_indices[0]]
    ], axis=1)

    # split neighborhood features by variable, along with the corresponding variable
    var_cuts = np.diff(edge_indices[1]).nonzero()[0]+1
    nbr_feats = np.split(nbr_feats, var_cuts)
    nbr_vars = np.split(edge_indices[1], var_cuts)
    assert all([all(vs[0] == vs) for vs in nbr_vars])
    nbr_vars = [vs[0] for vs in nbr_vars]

    # process candidate variable neighborhoods only
    for var, nbr_id, cand_id in zip(*np.intersect1d(nbr_vars, candidates, return_indices=True)):
        cand_states[cand_id, :] = np.concatenate([
            variable_features[var, :],
            nbr_feats[nbr_id].min(axis=0),
            nbr_feats[nbr_id].mean(axis=0),
            nbr_feats[nbr_id].max(axis=0)])

    cand_states[np.isnan(cand_states)] = 0

    return cand_states


def extract_khalil_variable_features(model, candidates, root_buffer):
    """
    Keep zero-width fallback for now.
    You can later switch to ecole.observation.Khalil2016 if needed.
    """
    return np.zeros((len(candidates), 0), dtype=np.float32)


def preprocess_variable_features(features, interaction_augmentation, normalization):
    """
    Features preprocessing following Khalil et al. (2016) Learning to Branch in Mixed Integer Programming.

    Parameters
    ----------
    features : 2D np.ndarray
        The candidate variable features to preprocess.
    interaction_augmentation : bool
        Whether to augment features with 2-degree interactions (useful for linear models such as SVMs).
    normalization : bool
        Wether to normalize features in [0, 1] (i.e., query-based normalization).

    Returns
    -------
    variable_features : 2D np.ndarray
        The preprocessed variable features.
    """
    # 2-degree polynomial feature augmentation
    if interaction_augmentation:
        interactions = (
            np.expand_dims(features, axis=-1) * \
            np.expand_dims(features, axis=-2)
        ).reshape((features.shape[0], -1))
        features = np.concatenate([features, interactions], axis=1)

    # query-based normalization in [0, 1]
    if normalization:
        features -= features.min(axis=0, keepdims=True)
        max_val = features.max(axis=0, keepdims=True)
        max_val[max_val == 0] = 1
        features /= max_val

    return features


def load_flat_samples(filename, feat_type, label_type, augment_feats, normalize_feats):
    with gzip.open(filename, 'rb') as file:
        sample = pickle.load(file)

    state, khalil_state, best_cand, cands, cand_scores = sample['data']

    cands = np.array(cands)
    cand_scores = np.array(cand_scores)

    cand_states = []
    if feat_type in ('all', 'gcnn_agg'):
        cand_states.append(compute_extended_variable_features(state, cands))
    if feat_type in ('all', 'khalil'):
        cand_states.append(khalil_state)
    cand_states = np.concatenate(cand_states, axis=1)

    best_cand_idx = np.where(cands == best_cand)[0][0]

    # feature preprocessing
    cand_states = preprocess_variable_features(cand_states, interaction_augmentation=augment_feats, normalization=normalize_feats)

    if label_type == 'scores':
        cand_labels = cand_scores

    elif label_type == 'ranks':
        cand_labels = np.empty(len(cand_scores), dtype=int)
        cand_labels[cand_scores.argsort()] = np.arange(len(cand_scores))

    elif label_type == 'bipartite_ranks':
        # scores quantile discretization as in
        # Khalil et al. (2016) Learning to Branch in Mixed Integer Programming
        cand_labels = np.empty(len(cand_scores), dtype=int)
        cand_labels[cand_scores >= 0.8 * cand_scores.max()] = 1
        cand_labels[cand_scores < 0.8 * cand_scores.max()] = 0

    else:
        raise ValueError(f"Invalid label type: '{label_type}'")

    return cand_states, cand_labels, best_cand_idx

def load_flat_samples_modified(filename, feat_type, label_type, augment_feats, normalize_feats):
    """
    Modifies the `load_flat_samples` to adapt to the new structure in samples.
    """
    with gzip.open(filename, 'rb') as file:
        sample = pickle.load(file)

    # root data
    if sample['type'] == "root":
        state, khalil_state, cands, best_cand, cand_scores = sample['root_state'] # best_cand is relative to cands (in practical_l2b/02_generate_dataset.py)
        best_cand_idx = best_cand
    else:
        # data for gcnn
        obss, best_cand, obss_feats, _ = sample['obss']
        v, gcnn_c_feats, gcnn_e = obss
        gcnn_v_feats = v[:, :19] # gcnn features

        state = {'values':gcnn_c_feats}, gcnn_e, {'values':gcnn_v_feats}
        sample_cand_scores = obss_feats['scores']
        cands = np.where(sample_cand_scores != -1)[0]
        cand_scores = sample_cand_scores[cands]
        khalil_state = v[:,19:-1][cands]

        best_cand_idx = np.where(cands == best_cand)[0][0]


    cands = np.array(cands)
    cand_scores = np.array(cand_scores)

    cand_states = []
    if feat_type in ('all', 'gcnn_agg'):
        cand_states.append(compute_extended_variable_features(state, cands))
    if feat_type in ('all', 'khalil'):
        cand_states.append(khalil_state)
    cand_states = np.concatenate(cand_states, axis=1)
    # best_cand_idx = np.where(cands == best_cand)[0][0]

    # feature preprocessing
    cand_states = preprocess_variable_features(cand_states, interaction_augmentation=augment_feats, normalization=normalize_feats)

    if label_type == 'scores':
        cand_labels = cand_scores

    elif label_type == 'ranks':
        cand_labels = np.empty(len(cand_scores), dtype=int)
        cand_labels[cand_scores.argsort()] = np.arange(len(cand_scores))

    elif label_type == 'bipartite_ranks':
        # scores quantile discretization as in
        # Khalil et al. (2016) Learning to Branch in Mixed Integer Programming
        cand_labels = np.empty(len(cand_scores), dtype=int)
        cand_labels[cand_scores >= 0.8 * cand_scores.max()] = 1
        cand_labels[cand_scores < 0.8 * cand_scores.max()] = 0

    else:
        raise ValueError(f"Invalid label type: '{label_type}'")

    return cand_states, cand_labels, best_cand_idx
