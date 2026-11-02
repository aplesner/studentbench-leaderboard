"""Reproduce the paper's AI quartiles and extend both estimators to human tutors.

Requires numpy, pandas, scipy, matplotlib. Run from any directory; optional
--paper-root verifies the reproduced AI results against the existing figure data.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/studentbench-quartiles-mpl')
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/human-quartiles-20260918'
CUTS = {'quant': (9, 13, 18), 'verbal': (9, 12, 15)}
LABELS = {'quant': ['≤9', '10–13', '14–17', '≥18'],
          'verbal': ['≤9', '10–12', '13–14', '≥15']}


def quartile(score, section):
    a, b, c = CUTS[section]
    return 'Q' + str(1 if score <= a else 2 if score <= b else 3 if score < c else 4)


def contrast(block, arm, adjusted):
    a = block.loc[block.arm == arm, 'gain_pp'].to_numpy()
    c = block.loc[block.arm == 'control', 'gain_pp'].to_numpy()
    assert min(len(a), len(c)) > 1
    if adjusted:
        x = np.column_stack([np.ones(len(block)), block.pre_pp, (block.arm == arm).astype(float)])
        y = block.post_pp.to_numpy()
        assert np.linalg.matrix_rank(x) == 3
        inv = np.linalg.inv(x.T @ x)
        beta = inv @ x.T @ y
        residual = y - x @ beta
        leverage = np.sum((x @ inv) * x, axis=1)
        assert np.all(leverage < 1)
        scaled = residual / (1 - leverage)
        cov = inv @ (x.T @ ((scaled ** 2)[:, None] * x)) @ inv
        estimate, se = beta[2], np.sqrt(cov[2, 2])
        critical = stats.norm.ppf(.975)
        p = 2 * stats.norm.sf(abs(estimate / se))
        df = len(block) - 3
    else:
        va, vc = a.var(ddof=1) / len(a), c.var(ddof=1) / len(c)
        estimate, se = a.mean() - c.mean(), np.sqrt(va + vc)
        df = (va + vc) ** 2 / (va ** 2 / (len(a)-1) + vc ** 2 / (len(c)-1))
        critical = stats.t.ppf(.975, df)
        p = 2 * stats.t.sf(abs(estimate / se), df)
    return dict(n_tutor=len(a), n_control=len(c), tutor_mean_gain_pp=a.mean(),
                control_mean_gain_pp=c.mean(), gain_over_control_pp=estimate,
                ci_low_pp=estimate-critical*se, ci_high_pp=estimate+critical*se,
                se_pp=se, df=df, p_raw=p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paper-root', type=Path)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    records, hashes = [], []
    source = ROOT / 'studentbench_data_ssot/1_main_leaderboard_data'
    for path in sorted(source.glob('*/*/task_id_*/student_data/student.json')):
        payload = path.read_bytes()
        d = json.loads(payload)
        pre, post = d['pretest_score_points'], d['posttest_score_points']
        assert post - pre == d['score_lift_points']
        assert d['assessment_max_score_points'] == 27
        section = d['instrument_id'].removeprefix('gre_')
        records.append(dict(student_id=d['student_id'], section=section, arm=d['tutoring_arm'],
                            pre=pre, post=post, pre_pp=100*pre/27, post_pp=100*post/27,
                            gain_pp=100*(post-pre)/27, quartile=quartile(pre, section)))
        hashes.append(dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(payload).hexdigest()))
    data = pd.DataFrame(records)
    assert data.student_id.is_unique
    assert len(data) == 2469
    expected = {'quant': {'ai':1139, 'control':91, 'human':61},
                'verbal': {'ai':1000, 'control':99, 'human':79}}
    for section, counts in expected.items():
        block = data[data.section == section]
        assert block.arm.value_counts().to_dict() == counts
        np.testing.assert_array_equal(np.quantile(block.pre, [.25,.5,.75]), CUTS[section])
    rows = []
    for section in CUTS:
        for i in range(4):
            q = f'Q{i+1}'
            for arm in ['ai', 'human']:
                block = data[(data.section == section) & (data.quartile == q) & data.arm.isin([arm,'control'])]
                for adjusted in [False, True]:
                    rows.append(dict(section=section, quartile=q, pretest_range=LABELS[section][i],
                                     tutor=arm, estimator='ANCOVA HC3' if adjusted else 'raw Welch',
                                     **contrast(block, arm, adjusted)))
    result = pd.DataFrame(rows)
    # Same eight-comparison Holm family as the paper, separately per tutor and estimator.
    for _, group in result.groupby(['tutor','estimator']):
        order = group.p_raw.sort_values().index
        adjusted = np.minimum(1, np.maximum.accumulate(result.loc[order,'p_raw'].to_numpy()*np.arange(8,0,-1)))
        result.loc[order,'p_holm_8'] = adjusted
    verification = dict(cohort_counts=expected, cutoffs=CUTS, unique_students=True,
                        stored_gain_identity_verified=True, ai_reference_verified=False)
    if args.paper_root:
        ref = args.paper_root / 'figures/revised_learning/data/proficiency_quartile_ai_control_contrasts.csv'
        original = pd.read_csv(ref)
        merged = result[result.tutor == 'ai'].merge(original, on=['section','quartile','estimator'], suffixes=('', '_ref'), validate='one_to_one')
        assert len(merged) == 16
        errors = {}
        for a,b in [('gain_over_control_pp','ai_minus_control_pp'),('ci_low_pp','ci_low_pp_ref'),
                    ('ci_high_pp','ci_high_pp_ref'),('p_holm_8','p_holm_across_8_quartile_contrasts'),
                    ('n_tutor','n_ai'),('n_control','n_control_ref')]:
            np.testing.assert_allclose(merged[a],merged[b],atol=1e-9,rtol=1e-9)
            errors[a] = float(np.max(np.abs(merged[a]-merged[b])))
        verification.update(ai_reference_verified=True, ai_max_absolute_errors=errors,
                            reference=str(ref), reference_sha256=hashlib.sha256(ref.read_bytes()).hexdigest())
    result.to_csv(OUT/'quartile_contrasts.csv',index=False)
    data.to_csv(OUT/'participant_scores.csv',index=False)
    (OUT/'source_manifest.json').write_text(json.dumps(hashes,indent=2)+'\n')
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    plot(result)
    human = result[(result.tutor=='human') & (result.estimator=='ANCOVA HC3')]
    print(human[['section','quartile','n_tutor','n_control','gain_over_control_pp','ci_low_pp','ci_high_pp','p_holm_8']].to_string(index=False))
    print('AI reference verified:',verification['ai_reference_verified'])


def plot(result):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':14,
                         'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False})
    fig, axes = plt.subplots(2,2,figsize=(14,10),sharex=True)
    fig.subplots_adjust(left=.16,right=.975,top=.84,bottom=.19,wspace=.48,hspace=.4)
    fig.suptitle('Learning gains over no tutoring, by starting proficiency',fontsize=21,x=.05,ha='left',y=.975,weight='bold')
    fig.text(.05,.932,'Same Q1–Q4 pre-test score groups as the current AI analysis · 27-question assessments',fontsize=12,color='#56616c')
    for r,arm in enumerate(['ai','human']):
        for col,section in enumerate(['quant','verbal']):
            ax=axes[r,col]
            d=result[(result.tutor==arm)&(result.section==section)]
            ax.axvline(0,color='#8a9299',lw=1)
            for estimator,offset,color,marker,label in [('raw Welch',-.13,'#89949d','o','Observed'),
                    ('ANCOVA HC3',.13,'#087ca7','s','Baseline-adjusted')]:
                b=d[d.estimator==estimator].sort_values('quartile')
                y=np.arange(4)+offset
                ax.errorbar(b.gain_over_control_pp,y,
                    xerr=[b.gain_over_control_pp-b.ci_low_pp,b.ci_high_pp-b.gain_over_control_pp],
                    fmt=marker,color=color,elinewidth=1.8,capsize=4,markersize=6,label=label)
            b=d[d.estimator=='ANCOVA HC3'].sort_values('quartile')
            ax.set_yticks(range(4),[f'{v.quartile}  {v.pretest_range}/27\nn={v.n_tutor} tutor, {v.n_control} control' for v in b.itertuples()])
            ax.tick_params(axis='y',length=0,labelsize=10,pad=10)
            ax.set_ylim(3.55,-.6)
            ax.set_title(f'{chr(65+r*2+col)}  {"AI tutors" if arm=="ai" else "Human tutors"} · {"Quantitative" if section=="quant" else "Verbal"}',loc='left',pad=16,weight='bold')
            ax.grid(axis='x',alpha=.15)
            ax.set_axisbelow(True)
            ax.set_xlim(-15,30)
            ax.tick_params(axis='x',labelbottom=True)
            if r==1: ax.set_xlabel('Tutor − no tutor learning gain (percentage points)',labelpad=10)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper left',bbox_to_anchor=(.045,.906),ncol=2,frameon=False)
    fig.text(.05,.058,'Points and bars: estimated differences and pointwise 95% CIs. Observed: Welch; adjusted: within-group linear ANCOVA, HC3.\nAI pools all 13 models by student. Human panels reuse the same control students. Intervals are not clustered by human tutor.',fontsize=10,color='#56616c',linespacing=1.7)
    for extension in ['png','svg']:
        fig.savefig(OUT/f'ai_human_quartiles.{extension}',dpi=180,facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
