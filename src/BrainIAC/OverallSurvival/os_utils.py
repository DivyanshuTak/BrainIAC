import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings


def plot_km_curve(data, time="survival_time", event="deadstatus_event", group="group", time_limit=4):
    df = data.copy()
    df.loc[df[time] >= time_limit, event] = 0  # Cap events at time_limit years
    T, E, G = df[time], df[event], df[group]

    kmfs = {}
    for grp in sorted(df[group].unique()):
        kmf = KaplanMeierFitter().fit(T[df[group] == grp], E[df[group] == grp], label=grp)
        kmfs[grp] = kmf

    results = multivariate_logrank_test(T, G, event_observed=E)
    
    fig, ax = plt.subplots(figsize=(5, 5))
    for grp, kmf in kmfs.items():
        kmf.plot(ax=ax, show_censors=True)

    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Survival Probability")
    ax.set_xlim(0, time_limit)
    ax.legend()

    p_val = results.p_value
    ax.text(0.7, 0.9, f"p = {p_val:.3f}", transform=ax.transAxes, fontsize=10)

    plt.tight_layout()
    return fig

def calculate_metrics(pred_probs, pred_labels, true_labels):
    """
    classification metrics.
    Args:
        pred_probs (numpy.ndarray): Predicted probabilities
        pred_labels (numpy.ndarray): Predicted labels
        true_labels (numpy.ndarray): Ground truth labels
        
    Returns:
        dict: Dictionary containing accuracy, precision, recall, F1, and AUC metrics
    """
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    try:
        auc = roc_auc_score(true_labels, pred_probs)
    except ValueError as e:
        warnings.warn(str(e))
        auc = float("nan")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }