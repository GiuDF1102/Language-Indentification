def FromRowToColumn(v):
    return v.reshape((v.size, 1))

def calcmean(D):
    return D.mean(1) #ritorno media sulle colonne