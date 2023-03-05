import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

#################################################################################################

def compute_metrics_over_thresholds(preds, gts, thresholds = np.linspace(0, 1, 101), eps = 1e-3):
    f1scores = []
    precisions=[]
    recalls=[]
    for t in thresholds:
        predict = (preds > t).astype(np.float32)

        tp = ((predict >= 0.5) & (gts >= 0.5)).sum()
        fp = ((predict >= 0.5) & (gts < 0.5)).sum()
        fn = ((predict < 0.5) & (gts >= 0.5)).sum()

        r = tp / (tp + fn + eps)
        p = tp / (tp + fp + eps)
        f1 = 2 * r * p / (r + p + eps)
        f1scores.append(f1)
        precisions.append(p)
        recalls.append(r)
    f1scores = np.array(f1scores)
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    return f1scores, precisions, recalls, thresholds    


def compute_best_metrics(cancer_p, cancer_t):

    fpr, tpr, thresholds = metrics.roc_curve(cancer_t, cancer_p)
    auc = metrics.auc(fpr, tpr)

    f1scores, precisions, recalls, thresholds = compute_metrics_over_thresholds(cancer_p, cancer_t)
    i = f1scores.argmax()
    f1score, precision, recall, threshold = f1scores[i], precisions[i], recalls[i], thresholds[i]

    specificity = ((cancer_p < threshold ) & ((cancer_t <= 0.5))).sum() / (cancer_t <= 0.5).sum()
    sensitivity = ((cancer_p >= threshold) & ((cancer_t >= 0.5))).sum() / (cancer_t >= 0.5).sum()

    return {
        'auc': auc,
        'threshold': threshold,
        'f1score': f1score,
        'precision': precision,
        'recall': recall,
        'sensitivity': sensitivity,
        'specificity': specificity,
    }


def compute_pfbeta(labels, predictions, beta=1):
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(labels)):
        prediction = min(max(predictions[idx], 0), 1)
        if (labels[idx]):
            y_true_count += 1
            ctp += prediction
            #cfp += 1 - prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp+1e-8)
    c_recall = ctp / y_true_count
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0


def print_all_metric(valid_df):

	print(f'{"    ": <16}    \tauc      @th     f1      | 	prec    recall  | 	sens    spec ')
	for site_id in [0,1,2]:
		if site_id>0:
			site_df = valid_df[valid_df.site_id == site_id].reset_index(drop=True)
		else:
			site_df = valid_df
		# ---

		gb = site_df
		m = compute_best_metrics(gb.cancer_p, gb.cancer_t)
		text = f'{"single image": <16} [{site_id}]'
		text += f'\t{m["auc"]:0.5f}'
		text += f'\t{m["threshold"]:0.5f}'
		text += f'\t{m["f1score"]:0.5f} | '
		text += f'\t{m["precision"]:0.5f}'
		text += f'\t{m["recall"]:0.5f} | '
		text += f'\t{m["sensitivity"]:0.5f}'
		text += f'\t{m["specificity"]:0.5f}'
		#text += '\n'
		print(text)


		# ---

		gb = site_df[['patient_id', 'laterality', 'cancer_t', 'cancer_p']].groupby(['patient_id', 'laterality']).mean()
		m = compute_best_metrics(gb.cancer_p, gb.cancer_t)
		text = f'{"grouby mean()": <16} [{site_id}]'
		text += f'\t{m["auc"]:0.5f}'
		text += f'\t{m["threshold"]:0.5f}'
		text += f'\t{m["f1score"]:0.5f} | '
		text += f'\t{m["precision"]:0.5f}'
		text += f'\t{m["recall"]:0.5f} | '
		text += f'\t{m["sensitivity"]:0.5f}'
		text += f'\t{m["specificity"]:0.5f}'
		#text += '\n'
		print(text)

		# ---
		gb = site_df[['patient_id', 'laterality', 'cancer_t', 'cancer_p']].groupby(['patient_id', 'laterality']).max()
		m = compute_best_metrics(gb.cancer_p, gb.cancer_t)
		text = f'{"grouby max()": <16} [{site_id}]'
		text += f'\t{m["auc"]:0.5f}'
		text += f'\t{m["threshold"]:0.5f}'
		text += f'\t{m["f1score"]:0.5f} | '
		text += f'\t{m["precision"]:0.5f}'
		text += f'\t{m["recall"]:0.5f} | '
		text += f'\t{m["sensitivity"]:0.5f}'
		text += f'\t{m["specificity"]:0.5f}'
		#text += '\n'
		print(text)
		print(f'--------------\n')


def compute_all(df, plot_save_path):
    print(f'Saving plot to {plot_save_path}')
    df['cancer_p'] = df['preds']
    df['cancer_t'] = df['targets']
    print_all_metric(df)


    gb = df[['site_id', 'patient_id','laterality', 'cancer_t', 'cancer_p']].groupby(['patient_id', 'laterality']).mean()
    gb.loc[:, 'cancer_t'] = gb.cancer_t.astype(int)
    m = compute_best_metrics(gb.cancer_p, gb.cancer_t)
    text = f'{"grouby mean()": <16}'
    text += f'\t{m["auc"]:0.5f}'
    text += f'\t{m["threshold"]:0.5f}'
    text += f'\t{m["f1score"]:0.5f} | '
    text += f'\t{m["precision"]:0.5f}'
    text += f'\t{m["recall"]:0.5f} | '
    text += f'\t{m["sensitivity"]:0.5f}'
    text += f'\t{m["specificity"]:0.5f}'
    text += '\n'
    print(text)

    pfbeta = compute_pfbeta(gb.cancer_t.values, gb.cancer_p.values, beta=1)
    print('PROBABILITY-FBETA:',pfbeta)
    
    plot_pr_curve(gb, plot_save_path)


   


def plot_pr_curve(df, plot_save_path):
    f1scores, precisions, recalls, thresholds = compute_metrics_over_thresholds(df.cancer_p, df.cancer_t)
    i = f1scores.argmax()
    f1score_max, precision_max, recall_max, threshold_max = f1scores[i], precisions[i], recalls[i], thresholds[i]
    print(f'f1score_max = {f1score_max}, precision_max = {precision_max}, recall_max = {recall_max}, threshold_max = {threshold_max}')

    _, axs = plt.subplots(2, 2, figsize=(20, 15))

    ############################################################################
    ### PRECISION-RECALL CURVE
    f_scores = [0.2,0.3,0.4,0.5,0.6,0.7,0.8] #np.linspace(0.2, 0.8, num=8)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = axs[0, 0].plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        axs[0, 0].annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))
    axs[0, 0].plot([0,1],[0,1], color="gray", alpha=0.2)

    # overall
    precision, recall, threshold = metrics.precision_recall_curve(df.cancer_t, df.cancer_p)
    auc = metrics.auc(recall, precision)
    axs[0, 0].plot(recall,precision)
    s=axs[0, 0].scatter(recall[:-1],precision[:-1],c=threshold,cmap='hsv')
    axs[0, 0].scatter(recall_max, precision_max,s=30,c='k')

    # for each site
    precision, recall, threshold = metrics.precision_recall_curve(df.cancer_t[df.site_id==1], df.cancer_p[df.site_id==1])
    axs[0, 0].plot(recall,precision, '--', label='site_id=1')
    precision, recall, threshold = metrics.precision_recall_curve(df.cancer_t[df.site_id==2], df.cancer_p[df.site_id==2])
    axs[0, 0].plot(recall,precision, '--', label='site_id=2')

    axs[0, 0].set_xlim([0.0, 1.0])
    axs[0, 0].set_ylim([0.0, 1.05])

    text=''
    text+=f'MAX f1score {f1score_max: 0.5f} @ th = {threshold_max: 0.5f}\n'
    text+=f'prec {precision_max: 0.5f}, recall {recall_max: 0.5f}, pr-auc {auc: 0.5f}\n'

    axs[0, 0].legend()
    axs[0, 0].set_title(text)
    plt.colorbar(s,ax = axs[0, 0], label='threshold')
    axs[0, 0].set_xlabel('recall')
    axs[0, 0].set_ylabel('precision')


    ############################################################################
    # HISTOGRAM
    spacing=51

    for site_type in [0, 1, 2]:
        if site_type == 0:
            ax = axs[0, 1]
            sub_df = df
            title = 'All site'
        elif site_type == 1:
            ax = axs[1, 0]
            sub_df = df[df.site_id == site_type].reset_index(drop = True)
            title = 'Site 1'
        elif site_type == 2:
            ax = axs[1, 1]
            sub_df = df[df.site_id == site_type].reset_index(drop = True)
            title = 'Site 2'

        cancer_p = sub_df.cancer_p
        cancer_t = sub_df.cancer_t
        cancer_t = cancer_t.astype(int)
        pos, bin = np.histogram(cancer_p[cancer_t == 1], np.linspace(0, 1, spacing))
        neg, bin = np.histogram(cancer_p[cancer_t == 0], np.linspace(0, 1, spacing))
        pos = pos / (cancer_t == 1).sum()
        neg = neg / (cancer_t == 0).sum()
        # plt.plot(bin[1:],neg, alpha=1)
        # plt.plot(bin[1:],pos, alpha=1)
        bin = (bin[1:] + bin[:-1]) / 2
        ax.bar(bin, neg, width=1/spacing, label='neg', alpha=0.5)
        ax.bar(bin, pos, width=1/spacing, label='pos', alpha=0.5)
        ax.legend()
        ax.set_title(title)

    # plt.show()
    plt.savefig(plot_save_path)




'''


'''