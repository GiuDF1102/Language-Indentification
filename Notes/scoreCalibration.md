# To-Do
## MVG
- Effettuare la k-fold sui gaussiani e considerare come metrica la minDCF per i diversi EFFECTIVE priors
    - Fare ogni k-fold variando sul valore della dimensione della PCA

## Regressione Logistica
- Potrebbe essere necessario cambiare la funzione, essendo che le classi sono altamente sbilanciate, basta solamente 
- Effettuare la k-fold variando il valore della dimensione della PCA
    - per ogni dimensione della PCA sarebbe necessario vedere l'andamento del minDCF in funzione del parametro lambda, e dei diversi priors

## SVM
- Effettuare la k-fold variance il valore della dimensione della PCA
    - Lineare
        - per ogni dimensione della PCA sarebbe necessario vedere l'andamento del minDCF in funzione del parametro C e dei priors
    - Quadratica
        - per ogni dimensione della PCA sarebbe necessario vedere l'andamento del minDCF in funzione del parametro C e dei priors, ma ora si deve variare anche c (d rimane fisso a 2, essendo quadratico)
    - RBF
        - per ogni dimensione della PCA sarebbe necessario vedere l'andamento del minDCF in funzione del parametro C e dei priors, ma ora si deve variare anche gamma
