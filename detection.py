import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sgn
import scipy.stats as stats

#Importerer en matlab-fil fra en mappe som heter "data", som ligger i mappen som pythonfilen ligger i
#Tar seg av matlab sin filformat ved å fjerne metadata
def loadFile (filename):
    matfile = scipy.io.loadmat('./data/' + filename + '.mat')
    data = matfile[list(matfile)[-1]]
    return data


'''
Plotter normalfordelingen til de reelle og imaginære komponentene til OFDM modulerte signaler med normalfordeling og BPSK.
'''
def gaussianModel():
    gauss = loadFile('T1_data_Sk_Gaussian')
    bpsk = loadFile('T1_data_Sk_BPSK').T #Transponerer fordi rådataene til bpsk er feil vei

    #Regnet ut invers fouriertransform av signalene
    s_gauss = np.fft.ifft(gauss, axis=0)
    s_bpsk = np.fft.ifft(bpsk, axis = 0)

    #Hjelpefunksjon for plotting
    def plotGaussianHist(data):
        #Tilpasser en normalfordeling til histogrammet
        loc, scale = stats.norm.fit(data) 
        x = np.linspace(np.min(data), np.max(data), 100)
        #plotter histogrammet
        plt.hist(data,50, density=True)
        #plotter tilpassingen
        plt.plot(x, stats.norm.pdf(x,loc, scale))

    #plotter histogram for de reelle og imaginære delene
    plt.subplots_adjust(hspace=0.4)
    plt.subplot(2, 2, 1)
    plt.title("Normalfordelte symboler, reell")
    plotGaussianHist(np.real(s_gauss))

    plt.subplot(2, 2, 2)
    plt.title("Normalfordelte symboler, imaginære")
    y = np.imag(s_gauss)
    plotGaussianHist(y)

    plt.subplot(2, 2, 3)
    plt.title("BPSK symboler, reell")
    y = np.real(s_bpsk)
    plotGaussianHist(y)

    plt.subplot(2, 2, 4)
    plt.title("BPSK symboler, imaginære")
    y = np.imag(s_bpsk)
    plotGaussianHist(y)
    plt.show()

    #Estimerer forventningsveriene
    pow_gauss = np.mean(np.real(s_gauss)*np.imag(s_gauss))
    pow_bpsk = np.mean(np.real(s_bpsk)*np.imag(s_bpsk))
    mean_gauss = np.mean(np.real(s_gauss)) +  1j * np.mean(np.imag(s_gauss))
    mean_bpsk = np.mean(np.real(s_bpsk)) +  1j * np.mean(np.imag(s_bpsk))

    print("Komplekst gjennomsnitt, BPSK:",mean_bpsk, "Normalfordeling:", mean_gauss)
    print("Gjennomsnitt av komponentene multiplisert, BPSK:", pow_bpsk, "Normalfordeling:", pow_gauss)



def chiSquareModel():
    #importerer filene
    #Disse inneholder signaler der vi vet at det ene er en ledig kanal og den andre ikke er ledig
    x_H0 = loadFile('T3_data_x_H0')
    x_H1 = loadFile('T3_data_x_H1')
    #Dekomponerer i reell og imaginære deler
    x_r_H0 = np.real(x_H0)
    x_i_H0 = np.imag(x_H0)
    x_r_H1 = np.real(x_H1)
    x_i_H1 = np.imag(x_H1)
    #Estimerer variansen til støy og det sendte signalet ved å bruke datasett med rene signaler
    #Det vil si at disse filene inneholder signaler som kun er støy, eller kun er et signal
    var_s = np.var(loadFile('T3_data_sigma_s'))
    var_w = np.var(loadFile('T3_data_sigma_w'))

    print("Varians s:", var_s, "Varians w:", var_w)

    #Setter opp fordelingene som skal testes
    #Disse burde ha kji kvadratfordeling med 2 frihetsgrader
    H0 = 2 * (x_r_H0**2 + x_i_H0**2) / (var_s)
    H1 = 2 * (x_r_H1**2 + x_i_H1**2) / (var_s + var_w)

    x = np.linspace(0, 10, 100)

    bins = 100

    #Bruker 2 frihetsgrader for å finne fordelingen
    H0_pdf = stats.chi2.pdf(x, 2)
    H1_pdf = stats.chi2.pdf(x, 2)

    #plotter histogrammene og tilpassingene

    plt.subplot(2,1,1)
    plt.title("Histogram av treningsdata, H0")
    plt.xlim(0,14)
    plt.hist(H0, bins=bins, density=True)
    plt.plot(x, H0_pdf)
    plt.legend(["Kji-kvadrat tilpasset", "Histogram"])

    plt.subplot(2,1,2)
    plt.title("Histogram av treningsdata, H1")
    plt.xlim(0,14)
    plt.hist(H1, bins=bins, density=True)
    plt.plot(x, H1_pdf)
    plt.legend(["Kji-kvadrat tilpasset", "Histogram"])
    plt.show()


def ROC():
    #Estimerer variansen
    var_s = np.var(loadFile('T3_data_sigma_s'))
    var_w = np.var(loadFile('T3_data_sigma_w'))

    #Hjelpefunksjon for å regne ut n, grenseverdien for detektoren
    def n (l, K, var_w, var_s):
            return (np.log(l)*np.log(var_w) - np.log(var_s + var_w) )/((1/(var_s + var_w) - np.log(l)/var_w) ) 

    #Plotter ROC for ulike K
    for K in range(1,52,10):
        Pfa = np.linspace(0,1,1000)
        n = stats.chi2.isf(Pfa,df=K)*var_w/K
        Pd = stats.chi2.sf(K*n/((var_w + var_s)),df=K)
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.plot(Pfa,Pd, label=("K = " + str(K)))

    legend = ["K = " + str(i) for i in range(1,52,10)]
    plt.legend(legend)
    plt.xlabel("Sannsynlighet for falsk alarm")
    plt.ylabel("Sannsynlighet for deteksjon")
    plt.title("ROC av kji-kvadrat detektor")
    plt.show()

def compareDistributions():
    #Estimerer variansen
    var_s = np.var(loadFile('T3_data_sigma_s'))
    var_w = np.var(loadFile('T3_data_sigma_w'))

    #vilkårlige K og n, K bør være over 20 for at sentralgrenseteoremet skal passe bra nok
    n = np.linspace(0,6,1000)
    K = 50

    plt.xlabel("Terskel")
    plt.ylabel("Sannsynlighet")
    #Regner ut fordelingene til P_d og P_fa
    #Dette er den normalfordelte tilnærmingen
    Pfa = stats.norm.sf( (K*((n / var_w) - 1))/np.sqrt(2 * K))
    Pd = stats.norm.sf((K*((n / (var_w + var_s)) - 1))/np.sqrt(2*K))

    plt.plot(n,Pfa)
    plt.plot(n,Pd)

    #Dette er kji-kvadratfordelingen
    Pfa = stats.chi2.sf(K*n/(var_w), df=K)
    Pd = stats.chi2.sf(K*n/((var_w + var_s)), df=K)

    plt.plot(n,Pfa, linestyle="--")
    plt.plot(n,Pd, linestyle="--")

    plt.legend(("Falsk alarm, normalfordelt","Deteksjon, normalfordelt","Falsk alarm, kji-kvadrat","Deteksjon, kji-kvadrat") )
    plt.title("Sammenligning av kji-kvadrat-detektor og normalfordelt detektor")
    plt.show()

def complexity():
    #Estimerer variansen
    var_s = np.var(loadFile('T3_data_sigma_s'))
    var_w = np.var(loadFile('T3_data_sigma_w'))

    Pfa = 0.1
    Pda = np.linspace(0.95,0.5, 100)

    #hjelpefunksjon for å regne ut K
    def k (pfa,pda):
        iQpfa = stats.norm.isf(pfa)
        iQpda = stats.norm.isf(pda)
        return ( (-np.sqrt(2)*(iQpfa - iQpda*(var_w + var_s)/var_w) - np.sqrt(  2*(iQpfa - iQpda*(var_w + var_s)/var_w)**2  + 4  )   ) /2   )**2
    
    #plotter for ulike sannsynligheter for falsk alarm
    legend = []
    for fa in np.linspace(0.0001,0.05,5):
        
        pda = np.linspace(0.0001,0.9999,100) # stats.chi2.sf(K*ns/(var_w),df=K)

        #ns = n(np.linspace(0,10,1000), K, var_w, var_s)
        K = k(fa,pda)
        plt.plot(pda,K)
        legend.append("Pfa = " + str(round(fa*100, 2)) + "%")
    plt.legend(legend)
    plt.ylabel("Antall sampler, K")
    plt.xlabel("Sannsynlighet for deteksjon")
    plt.show()

def experiment():
    
    var_s = 5
    var_w = 1
    K = 256
    # importerer de 100 ulike realiseringene av signalet
    # disse 100 inneholder et signal på 256 sampler som er ledige kanaler eller ikke ledig
    signal = loadFile("T8_numerical_experiment")

    test = np.sum(np.square(np.abs(signal)), axis=0)/K

    # hjelpefunksjon for plotting av detektoreksperimentet
    # n er grensen for detektoren
    def plotDetector(data, n):
        plt.xlabel("Realiseringer")
        plt.ylabel("Gjennomsnittlig effekt")
        color_chi = np.where((data > n_chi) , 'blue', 'orange')
        i = np.where((data > n_chi))
        detected = data[i]
        plt.scatter(i, detected, color="blue", marker=".")
        i = np.where((data < n_chi))
        not_detected = data[i]
        plt.scatter(i, not_detected, color="orange", marker=".")
        plt.plot(np.arange(0,len(data)), np.ones(len(data))*n_chi)
        plt.legend(["Detektor", "Detektert", "Ikke detektert"])
    # Setter først sannsynligheten for falsk alarm til 10%
    Pfa = 0.1
    # Regner ut grensene til de to detektorene, en for kji-kvadratfordeling og en for den normalfordelte
    n_chi = stats.chi2.isf(Pfa,df=K) * var_w / K
    n_norm = var_w * (stats.norm.isf(Pfa)*np.sqrt(2 * K) + K)/K

    # Plotter de to
    plt.subplots_adjust(hspace=0.4)
    plt.subplot(2,1,1)
    plt.title("Kji-kvadrat NP detektor, Pfa = 0.1")
    plotDetector(test,n_chi)

    plt.subplot(2,1,2)
    plt.title("Normalfordelt NP detektor, Pfa = 0.1")
    plotDetector(test,n_norm)
    plt.show()

    # setter så Pfa til 1% og plotter på nytt
    Pfa = 0.01

    n_chi = stats.chi2.isf(Pfa,df=K) * var_w / K
    n_norm = var_w * (stats.norm.isf(Pfa)*np.sqrt(2 * K) + K)/K

    plt.subplots_adjust(hspace=0.4)
    plt.subplot(2,1,1)
    plt.title("Kji-kvadrat NP detektor, Pfa = 0.01")
    plotDetector(test,n_chi)

    plt.subplot(2,1,2)
    plt.title("Normalfordelt NP detektor, Pfa = 0.01")
    plotDetector(test,n_norm)
    plt.show()

if __name__ == "__main__":
    
    gaussianModel()
    chiSquareModel()
    ROC()
    compareDistributions()
    complexity()
    experiment()