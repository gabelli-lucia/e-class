import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Set the id
id = 'FFY'

# Load data from file
dati = np.genfromtxt("scatter - "+id+".txt", delimiter=',')
domande = np.loadtxt("Post questions.txt", dtype=str, delimiter='\t', encoding='utf-8')
numeri = [roba.split(' ')[0][1:] for roba in domande]
fig = plt.figure()
ax = fig.add_subplot(111)


#importanza = dati[:,2]
#variazione = dati[:,1] - dati[:,0]

importanza_uno = dati[:,2]
variazione_uno = dati[:,1] - dati[:,0]
#importanza_due = dati[:,4]
#variazione_due = dati[:,3] - dati[:,1]

#slope, intercept, r_value, p_value, std_err = stats.linregress(importanza, variazione)
#regression_line = slope * importanza + intercept

#pearson_r = np.corrcoef(importanza, variazione)[0, 1]
#n = len(importanza)
#se_r = np.sqrt((1 - pearson_r**2) / (n - 2))
#z = 0.5 * np.log((1 + pearson_r) / (1 - pearson_r))
#cohen_d = z * np.sqrt(2 / n)

#ax.plot(importanza, regression_line, color='red', label="Pearson's r = "+str(round(pearson_r,2))+',\nstd.err. = '+str(round(se_r,2))+", Cohen's d = "+str(round(cohen_d,2)), alpha = 0.5)

#ax.scatter(importanza, variazione)
ax.scatter(importanza_uno, variazione_uno)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

for i, testo in enumerate(numeri):
    #ax.annotate(testo, (importanza[i], variazione[i]))
    ax.annotate(testo, (importanza_uno[i], variazione_uno[i]))

#for i, testo in enumerate(numeri):
#    if (i != 9 and i != 10 and i != 19 and i != 0):
#        ax.annotate(testo, (importanza[i], variazione[i]))
#    elif i != 0:
#        ax.annotate(testo, (importanza[i], variazione[i]-0.0175))
#    else:
#        ax.annotate(testo, (importanza[i]-0.175/2.5, variazione[i]-0.0175))





ax.set_xlabel("Students' perceived importance")
ax.set_xlim(1,5)
#ax.set_ylim(-0.3,0.3)
ax.set_xticks(np.arange(1, 6, 1))
ax.set_xticklabels(['Strongly disagree', '','','','Strongly agree'])
ax.set_ylabel('Pre-Post variation in expert-like response')
#ax.legend(loc = 'upper left')
plt.savefig('Scatter - '+id+'.png')
#plt.savefig('Scatter - '+id+' First module'+'.png')
plt.close()

#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.scatter(importanza_due, variazione_due)
#ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
#for i, testo in enumerate(numeri):
#    if i != 18 and i != 20:
#        ax.annotate(testo, (importanza_due[i], variazione_due[i]))
#    else:
#        ax.annotate(testo, (importanza_due[i], variazione_due[i]-0.0175))
#ax.set_xlabel("Students' perceived importance")
#ax.set_xlim(1,5)
#ax.set_ylim(-0.3,0.3)
#ax.set_xticks(np.arange(1, 6, 1))
#ax.set_xticklabels(['Strongly disagree', '','','','Strongly agree'])
#ax.set_ylabel('Pre-Post variation in expert-like response')
#ax.legend(loc = 'upper left')
#plt.savefig('Scatter - '+id+'.png')
#plt.savefig('Scatter - '+id+' Second module'+'.png')
#plt.close()