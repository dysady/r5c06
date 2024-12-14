import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scipy.spatial import ConvexHull
import matplotlib as mpl
import matplotlib.cm as cm
import seaborn as sns
from mca import MCA
from adjustText import adjust_text

class ACPService:
    def __init__(self,df):
        # Standardisation des données
        self.X = StandardScaler().fit_transform(df)
        self.df = df
        self.pca = PCA(n_components=2)
        self.pca_res = self.pca.fit_transform(self.X)
        self.explained_variance = list(self.pca.explained_variance_ratio_)
        self.components = self.pca.components_

    def biplot_with_adjusted_labels(self, components=[0, 1], score=None, coeff=None,
                                    coeff_labels=None, score_labels=None, circle='T', bigdata=1000, 
                                    cat=None, cmap="viridis", density=True):
        """
        Crée un biplot avec ajustement des labels en fonction des composantes principales.
        
        :param components: Liste des indices des composantes principales à afficher
        :param score: Score (projections des données sur les composantes principales)
        :param coeff: Coefficients des composantes principales
        :param coeff_labels: Labels des coefficients
        :param score_labels: Labels des scores
        :param circle: 'T' pour afficher un cercle unitaire
        :param bigdata: Seuil pour afficher les données sous forme de nuage de points ou densité
        :param cat: Catégories pour colorer les points
        :param cmap: Colormap à utiliser
        :param density: Si True, affichera la densité des points
        """
        if isinstance(self.pca, PCA):
            coeff = np.transpose(self.pca.components_[components, :])
            score = self.pca_res[:, components]

            if isinstance(self.df, pd.DataFrame):
                coeff_labels = list(self.df.columns)

        if score is not None:
            x = score

        if x.shape[1] > 1:
            xs = x[:, 0]
            ys = x[:, 1]
        else:
            xs = x
            ys = y

        if len(xs) != len(ys):
            print("Warning ! x et y n'ont pas la même taille !")

        scalex = 1.0 / (xs.max() - xs.min())
        scaley = 1.0 / (ys.max() - ys.min())

        temp = (xs - xs.min())
        x_c = temp / temp.max() * 2 - 1

        temp = (ys - ys.min())
        y_c = temp / temp.max() * 2 - 1

        data = pd.DataFrame({"x_c": x_c, "y_c": y_c})
        print("Attention : pour des facilités d'affichage, les données sont centrées-réduites")

        if cat is None:
            cat = [0] * len(xs)
        elif len(pd.Series(cat)) == 1:
            cat = list(pd.Series(cat)) * len(xs)
        elif len(pd.Series(cat)) != len(xs):
            print("Warning ! Nombre anormal de catégories !")

        cat = pd.Series(cat).astype("category")

        fig = plt.figure(figsize=(10, 10), facecolor='w')
        ax = fig.add_subplot(111)

        # Affichage des points
        if len(xs) < bigdata:
            ax.scatter(x_c, y_c, c=cat.cat.codes, cmap=cmap)

            if density:
                print("Warning ! Le mode density actif n'apparait que si BigData est paramétré.")

        else:
            # Color
            norm = mpl.colors.Normalize(vmin=0, vmax=(len(np.unique(cat.cat.codes))))
            cmap = cmap
            m = cm.ScalarMappable(norm=norm, cmap=cmap)

            if density:
                sns.set_style("white")
                sns.kdeplot(x="x_c", y="y_c", data=data)

                if len(np.unique(cat)) <= 1:
                    sns.kdeplot(x="x_c", y="y_c", data=data, cmap="Blues", shade=True, thresh=0)
                else:
                    for i in np.unique(cat):
                        color_temp = m.to_rgba(i)
                        sns.kdeplot(x="x_c", y="y_c", data=data[cat == i], color=color_temp,
                                    shade=True, thresh=0.25, alpha=0.25)

            for cat_temp in cat.cat.codes.unique():
                x_c_temp = [x_c[i] for i in range(len(x_c)) if (cat.cat.codes[i] == cat_temp)]
                y_c_temp = [y_c[i] for i in range(len(y_c)) if (cat.cat.codes[i] == cat_temp)]

                points = np.array([x_c_temp, y_c_temp]).T
                hull = ConvexHull(points)

                for simplex in hull.simplices:
                    color_temp = m.to_rgba(cat_temp)
                    plt.plot(points[simplex, 0], points[simplex, 1], color=color_temp)

        if coeff is not None:
            if circle == 'T':
                x_circle = np.linspace(-1, 1, 100)
                y_circle = np.linspace(-1, 1, 100)
                X, Y = np.meshgrid(x_circle, y_circle)
                F = X ** 2 + Y ** 2 - 1.0
                plt.contour(X, Y, F, [0])

            texts = []  # Pour stocker les annotations ajustées
            n = coeff.shape[0]
            for i in range(n):
                plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5,
                        head_width=0.05, head_length=0.05)
                if coeff_labels is None:
                    label = "Var" + str(i + 1)
                else:
                    label = coeff_labels[i]

                # Ajouter les labels à ajuster
                texts.append(
                    ax.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, label, fontsize=10, ha='center', va='center')
                )

            # Ajustement des labels pour éviter les chevauchements
            adjust_text(
                texts,
                arrowprops=dict(arrowstyle='->',
                color='blue'),
                expand_text=(1.2, 1.5),  # Augmenter l'espace autour des labels
                expand_points=(1.2, 1.5),  # Espacement des points
                force_text=0.5,  # Priorité sur les labels
                force_points=0.3  # Priorité sur les points
                )

        plt.xlim(-1, 1.2)
        plt.ylim(-1, 1.2)
        plt.xlabel("PC{}".format(1))
        plt.ylabel("PC{}".format(2))
        plt.grid(linestyle='--')
        plt.show()


# Exemple d'utilisation
if __name__ == "__main__":
    df= pd.read_csv("number-of-deaths-by-risk-factor.csv")
    acp = ACPService(df)
    acp.biplot_with_adjusted_labels(
        score=acp.pca_res[:, 0:2],
        coeff=np.transpose(acp.components[0:2, :]),
        coeff_labels=df.columns,
        cat=acp.explained_variance[0:1],
        density=False
    )
