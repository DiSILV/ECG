## Results per subject
metric_to_print='Ratio'
ResultsMean=ResultsMeanDict[metric_to_print]
ResultsStd=ResultsStdDict[metric_to_print]

ResultsTabular=[]
for r_m,r_s in zip(ResultsMean,ResultsStd):
    ResultsTabular.append([r_m[0]]+[str("%.2f (%.2f)"%(s[0],s[1])) for s in zip(r_m[1:],r_s[1:])])
tab_per_subject=tabulate(ResultsTabular,headers=["Input","Ours"]+BaselinesToComare,tablefmt='latex',floatfmt=".2g")
print(tab_per_subject)

## Aggregated results
# sort according input
sory_by='Ratio'
sorted_inds=np.argsort(ResultsMeanDict[sory_by][:,1])[::-1]
SortedResults=np.array([ResultsMean[sorted_inds,x] for x in range(ResultsMean.shape[1])]).T

# sort according to each method
sorted_inds=np.argsort(ResultsMeanDict[sory_by],axis=0)[::-1]
SortedResults=np.array([ResultsMean[sorted_inds[:,x],x] for x in range(ResultsMean.shape[1])]).T

AggResultsTabular=[]
for k in [5,10,20,60]:
    curr_row=['Top %g'%k]+[str("%.2f (%.2f)"%(s[0],s[1])) for s in zip(np.mean(SortedResults[1:k,1:],axis=0),np.std(SortedResults[1:k,1:],axis=0))]
    AggResultsTabular.append(curr_row)
tab_agg=tabulate(AggResultsTabular,headers=["Input","Ours"]+BaselinesToComare,tablefmt='latex',floatfmt=".2g")
print(tab_agg)