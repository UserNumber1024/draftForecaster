from connector import MOEXAdapter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.manifold import TSNE
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas_ta as pta
from math import ceil

# CONSTANTS--------------------------------------------------------------------
clf = 'clf'
regr = 'regr'
file = 'file'
moex = 'moex'
# -----------------------------------------------------------------------------


class MOEXtoXY:
    """Get OHCL-data from MOEX and transform it to train/prediction dataset."""

    def __init__(self, moex_ticker_ids={}, ta_indicators=[]):
        """Initialize default data processing and preparation params.

        Parameters
        ----------
        moex_ticker_ids: dict of supported tickers
            uses for converting ticker into a numeric id,
            i.e. into a categorization attribute of the model

        ta_indicators: list of supported TA indicators

        """
        if moex_ticker_ids == {}:
            self.moex_ticker_ids = {'SBER': 0.1,
                                    'GAZP': 0.2,
                                    'LKOH': 0.3,
                                    'SIBN': 0.4
                                    }
        if ta_indicators == []:
            self.ta_indicators = [
                'tickerId',
                # 'macd12_26_9', отражает стоимость, т.е. завивит от тикера и даты, надо как-то нормализовать
                'rsi3',
                'rsi5',
                'rsi14',
                'rsi21',
                'wpr14',
                'stoch14_3',
                'adx14',
                'atr14',
                'cci14',
                # 'mom1',  какие-то выбросы, надо разобраться, может 24+- исключить из выборки (хотя лучше нет)
                # 'stdev15', порядок отклонения также зависит от порядка цены. попробовать ввести некий множитель для цен
                'bop',
                'trix14',
                # 'ema5_ohlc', отражает стоимость
                # 'ema10_ohlc', т.е. у тикеров разная на порядки
                # 'ema15_ohlc', можно попробовать рассчитывать процентное отклонение от средней цены свечи
                'volume1dInc',
                'volume3dInc',
                'volume5dInc',
                'numTrades1dInc',
                'numTrades3dInc',
                'numTrades5dInc',
                'open1dInc',
                'open3dInc',
                'open5dInc',
                'close2openInc',
                'high2openInc',
                'low2openInc',
                'dayOfWeek']

    def prepare_XY(self,
                   tickers=[],
                   start="2015-01-01",
                   end="2023-12-31",
                   store2file=True,
                   file_folder='data/',
                   file_postfix='_full_hist.csv',
                   source=file,
                   length=1,
                   Y_type=clf
                   ):
        """Prepare X and Y for analysis and forecasting.

        The source data can be extracted from files (from previos run`s)
        or obtained from MOEX (with the possibility of saving to a file).

        For forecasting, it is better to request data for about a month
        due to specific trading indicators calculation. (set store2file=False)

        X and Y keeped as class attributes (it's expensive, but whatever)

        Parameters
        ----------
        tickers: list of MOEX tickers for processing
        (in upcase, e.g. "SBER" (see getMoex_ticker_ids))
            if empty, used default list of supprted tickers (moex_ticker_ids)

        start, end: str, period in "YYYY-MM-DD" format
            default 2015..2023 (not used when source=file)

        store2file: Boolean, default True
            define, whether the "raw" trading data should be saved to a file

        file_folder, file_postfix: str
            file attributes for saving/retrieving trade data

        source: {'file', 'moex'}, default 'file', defines the method
        of obtaining trade data
            * from the 'moex' (MOEX API)
            * or from a 'file' (from a previous run)

        length: int, default 1
            is used to calculate Y (profit),
            determines N days for which potential profit is calculated

        Y_type: {"clf", "regr"}, default 'clf'
            solving classification or regression problem,
            i.e. determine if Y is a target class or a variable.

        Return
        ------
        X and Y to use by models.
        Also return prices - OHLCV from MOEX enriched with X and Y.

        """
        if tickers == []:
            tickers = list(self.moex_ticker_ids.keys())
        X = pd.DataFrame()
        Y = pd.DataFrame()
        prices = pd.DataFrame()
        self.length = length
        self.Y_type = Y_type
        if source == moex:
            iss = MOEXAdapter()

        for ticker in tickers:
            filename = file_folder + ticker + file_postfix
            if source == moex:
                ticker_data = iss.get_ticker_history(
                    ticker, start, end).reset_index(drop=True)
                if store2file:
                    ticker_data.to_csv(filename)
            elif source == file:
                ticker_data = pd.read_csv(filename)

            x_ticker, y_ticker, prices_ticker = self.__сalc_TA_profit(
                ticker, ticker_data)
            X = pd.concat([X, x_ticker])
            Y = pd.concat([Y, y_ticker])
            prices = pd.concat([prices, prices_ticker])
            X.reset_index(drop=True, inplace=True)
            Y.reset_index(drop=True, inplace=True)
            prices.reset_index(drop=True, inplace=True)

        self.X = X
        self.Y = Y
        return X, Y, prices

    def __get_OHLCV(self, moex_data):
        """Extract OHLCV columns from MOEX full data, rename to common ones."""
        prices = moex_data[["TRADEDATE", "OPEN", "LOW",
                            "HIGH", "LEGALCLOSEPRICE",
                           "VOLUME", "NUMTRADES", "SECID"]]
        prices.columns = ['Date', 'Open', 'Low',
                          'High', 'Close', 'Volume', 'Numtrades', 'Ticker']
        return prices

    def __сalc_TA_profit(self, ticker: str, moex_ticker_data: pd.DataFrame):
        """Generate indicators for model training and forecasting.

        length and Y_type retrieved from class attributes.

        Parameters
        ----------
        ticker: str
            MOEX ticker in upcase, e.g. "SBER" (see getMoex_ticker_ids),

        moex_data
            MOEX API-response in pd.DataFrame form,

        Return
        ------
        for concrete ticker:
            - X and Y to use by models.
            - prices - OHLCV from MOEX enriched with X and Y.
        """
        length = self.length
        Y_type = self.Y_type
        prices = self.__get_OHLCV(moex_ticker_data)

        # remove rows with OPEN=0
        # cause it looks like there was no trading that day
        prices = prices[prices.Open > 0]

        X = pd.DataFrame()

        # Split HLOС into pd.Series just for ease of use
        H = prices['High']
        L = prices['Low']
        Op = prices['Open']
        C = prices['Close']
        V = prices['Volume']
        N = prices['Numtrades']
        Date = pd.to_datetime(prices['Date'])

        for i in self.ta_indicators:
            if i == 'tickerId':
                # convert ticker into a numeric id,
                # i.e. into a categorization attribute of the model
                T = prices['Ticker']
                T = T.replace(self.moex_ticker_ids)
                T.name = 'TickerId'
                X = pd.concat([X, pd.DataFrame(T)])

            elif i == 'macd12_26_9':
                X = pd.concat([X, pta.macd(close=C,
                                           fast=12, slow=26, signal=9)],
                              axis=1).fillna(0).replace(np.inf, 0)

            elif i == 'rsi3':
                X = pd.concat([X, pta.rsi(close=C, length=3)/100],
                              axis=1).fillna(0).replace(np.inf, 0)

            elif i == 'rsi5':
                X = pd.concat([X, pta.rsi(close=C, length=5)/100],
                              axis=1).fillna(0).replace(np.inf, 0)

            elif i == 'rsi14':
                X = pd.concat([X, pta.rsi(close=C,
                                          length=14)/100],
                              axis=1).fillna(0).replace(np.inf, 0)

            elif i == 'rsi21':
                X = pd.concat([X, pta.rsi(close=C, length=21)/100],
                              axis=1).fillna(0).replace(np.inf, 0)
                # !!! ^^^^^^возможно есть смысл сделать по всем столбцам

            elif i == 'wpr14':
                X = pd.concat([X, pta.willr(high=H, low=L, close=C,
                                            length=14)/100],
                              axis=1).fillna(0).replace(np.inf, 0)
            elif i == 'stoch14_3':
                X = pd.concat([X, pta.stoch(high=H, low=L, close=C,
                                            k=14, d=3)/100],
                              axis=1).fillna(0).replace(np.inf, 0)

            elif i == 'adx14':
                X = pd.concat([X, pta.adx(high=H, low=L, close=C,
                                          length=14)/100],
                              axis=1).fillna(0).replace(np.inf, 0)
            elif i == 'atr14':
                X = pd.concat([X, pta.atr(high=H, low=L, close=C,
                                          length=14)/100],
                              axis=1).fillna(0).replace(np.inf, 0)
            elif i == 'cci14':
                X = pd.concat([X, pta.cci(high=H, low=L, close=C,
                                          length=14)/100],
                              axis=1).fillna(0).replace(np.inf, 0)
            elif i == 'mom1':
                X = pd.concat([X, pta.mom(close=C, length=1)],
                              axis=1).fillna(0).replace(np.inf, 0)
            # !!!^^^^^возможно есть смысл сделать по всем столбцам

            elif i == 'stdev15':
                X = pd.concat([X, pta.stdev(close=C, length=15)],
                              axis=1).fillna(0).replace(np.inf, 0)
            # !!!^^^^^^возможно есть смысл сделать по всем столбцам

            elif i == 'bop':
                X = pd.concat([X, pta.bop(open_=Op, high=H, low=L, close=C)],
                              axis=1).fillna(0).replace(np.inf, 0)
            elif i == 'trix14':
                X = pd.concat([X, pta.trix(close=C, length=14)],
                              axis=1).fillna(0).replace(np.inf, 0)
            # !!!^^^^^^возможно есть смысл сделать по всем столбцам

            elif i == 'ema5_ohlc':
                X = pd.concat([X, pta.ema(pta.ohlc4(Op, H, L, C), length=5)],
                              axis=1).fillna(0).replace(np.inf, 0)

            elif i == 'ema10_ohlc':
                X = pd.concat([X, pta.ema(pta.ohlc4(Op, H, L, C), length=10)],
                              axis=1).fillna(0).replace(np.inf, 0)

            elif i == 'ema15_ohlc':
                X = pd.concat([X, pta.ema(pta.ohlc4(Op, H, L, C), length=15)],
                              axis=1).fillna(0).replace(np.inf, 0)

            elif i == 'volume1dInc':
                X = pd.concat([X, pd.Series((V - V.shift(1)) /
                                            V.shift(1), name=i)],
                              axis=1).fillna(0).replace(np.inf, 0)

            elif i == 'volume3dInc':
                X = pd.concat([X, pd.Series((V - V.shift(3)) /
                                            V.shift(3), name=i)],
                              axis=1).fillna(0).replace(np.inf, 0)

            elif i == 'volume5dInc':
                X = pd.concat([X, pd.Series((V - V.shift(5)) /
                                            V.shift(5), name=i)],
                              axis=1).fillna(0).replace(np.inf, 0)

            elif i == 'numTrades1dInc':
                X = pd.concat([X, pd.Series((N - N.shift(1)) /
                                            N.shift(1), name=i)],
                              axis=1).fillna(0).replace(np.inf, 0)

            elif i == 'numTrades3dInc':
                X = pd.concat([X, pd.Series((N - N.shift(3)) /
                                            N.shift(3), name=i)],
                              axis=1).fillna(0).replace(np.inf, 0)

            elif i == 'numTrades5dInc':
                X = pd.concat([X, pd.Series((N - N.shift(5)) /
                                            N.shift(5), name=i)],
                              axis=1).fillna(0).replace(np.inf, 0)

            elif i == 'open1dInc':
                X = pd.concat([X, pd.Series((Op - C.shift(1)) /
                                            C.shift(1), name=i)],
                              axis=1).fillna(0).replace(np.inf, 0)

            elif i == 'open3dInc':
                X = pd.concat([X, pd.Series((Op - C.shift(3)) /
                                            C.shift(3), name=i)],
                              axis=1).fillna(0).replace(np.inf, 0)

            elif i == 'open5dInc':
                X = pd.concat([X, pd.Series((Op - C.shift(5)) /
                                            C.shift(5), name=i)],
                              axis=1).fillna(0).replace(np.inf, 0)

            elif i == 'close2openInc':
                X = pd.concat([X, pd.Series((C - Op) / Op, name=i)],
                              axis=1).fillna(0).replace(np.inf, 0)

            elif i == 'high2openInc':
                X = pd.concat([X, pd.Series((H - Op) / Op, name=i)],
                              axis=1).fillna(0).replace(np.inf, 0)

            elif i == 'low2openInc':
                X = pd.concat([X, pd.Series((L - Op) / Op, name=i)],
                              axis=1).fillna(0).replace(np.inf, 0)
            elif i == 'dayOfWeek':

                X = pd.concat([X, pd.Series(Date.dt.dayofweek/10,
                                            name='DayOfW')], axis=1)

        # Y - profit
        # if buy now, will be there an average profit after N days?
        # assume, that the profit should be at least 1 percent from transaction
        # calculate the mean price increment for N days
        mean = pta.hlc3(H, L, C).fillna(-1000).replace(np.inf, -1000)
        mean_forecast = pta.hlc3(
            H.shift(-length),  L.shift(-length),
            C.shift(-length)).fillna(-1000).replace(np.inf, -10)

        profit = pd.Series(((mean_forecast - mean) / mean), name='profitInc'
                           ).fillna(-1000).replace(np.inf, -1000)

        if Y_type == clf:
            Y = pd.Series((profit >= 0.01), name='profit').astype(int)
        elif Y_type == regr:
            Y = pd.Series(profit, name='profit')

        prices = pd.concat([prices, X, profit, Y, mean],
                           axis=1).reindex(
            prices.index).fillna(-1000).replace(np.inf, -1000)

        # drop last N=length rows, because Y not valid there
        X.drop(X.tail(length).index, inplace=True)
        Y.drop(Y.tail(length).index, inplace=True)
        prices.drop(prices.tail(length).index, inplace=True)

        return X, Y, prices

    def draw_X_Y(self):
        """Visualisation of training dataset."""
        h = ceil(len(self.X.columns)**(1/2))
        fig_hist = plt.figure(figsize=(30, 30))
        for i, column in enumerate(self.X.columns):
            plt.subplot(h, h, i + 1)
            sns.histplot(data=self.X[column], kde=True)
            plt.axvline(self.X[column].mean(), color='green', linestyle='--')
            plt.axvline(self.X[column].median(), color='black', linestyle='--')

        plt.tight_layout()
        plt.show()
        fig_hist.savefig("graph/x_hist.png")

        fig_box = plt.figure(figsize=(90, 90))
        sns.boxplot(data=self.X, palette="Set1", showmeans=True, orient='h')
        # plt.xticks(rotation=90)
        plt.show()
        fig_box.savefig("graph/x_box.png")

        fig_heat = plt.figure(figsize=(30, 30), dpi=80)
        sns.heatmap(self.X.corr(), cmap='RdYlGn', annot=True)
        plt.show()
        fig_heat.savefig("graph/x_heatmap.png")

        fig_Y = plt.figure(figsize=(10, 10))
        sns.histplot(data=self.Y[0])
        fig_Y.savefig("graph/y_hist.png")

    def draw_PCA(self, reduce_to_95=True, draw_expl=True,
                 draw_pca=True):
        """Draws training dataset, reduced to 3D projection with PCA.

        Also used in draw_tSNE method to reduce dimension of dataset.

        Parameters
        ----------
        reduce_to_95: Bool, default = True
            reduce dimension of dataset enought to provide
            95% level of explained variance

        draw_expl: Bool, default = True
            plot axplained variance vs dimension of dataset

        draw_pca: Bool, default = True
            scatter plot of PCA 3D projection

        """
        if self.Y_type == regr:
            return None

        if draw_expl:
            pca_expl = decomposition.PCA().fit(self.X)
            explained_variance_ratio = pca_expl.explained_variance_ratio_
            cumulative_variance = explained_variance_ratio.cumsum()

            fig_expl, ax_expl = plt.subplots(
                nrows=1, ncols=2, figsize=(10, 10))
            plt.title('Principal components number vs explained variance')

            ax_expl[0].plot(range(1, len(explained_variance_ratio) + 1),
                            explained_variance_ratio, marker='o')
            ax_expl[0].set_xlabel("Components number")
            ax_expl[0].set_ylabel("Explained variance")

            ax_expl[1].plot(range(1, len(cumulative_variance) + 1),
                            cumulative_variance, marker='o')
            ax_expl[1].set_xlabel("Components number")
            ax_expl[1].set_ylabel("Cumulative explained variance")

            fig_expl.tight_layout()
            plt.show()
            fig_expl.savefig("graph/pca_explain.png")

        if reduce_to_95:
            pca_95 = decomposition.PCA(0.95)
            X_reduced = pca_95.fit_transform(self.X)
        else:
            X_reduced = self.X

        if draw_pca:
            pca = decomposition.PCA(n_components=3)
            X_3d = pca.fit_transform(X_reduced)

            fig_draw = plt.figure(figsize=[15, 15])
            dimens = X_reduced.shape[1]
            plt.title(f"PCA {dimens}D data to 3D projection")

            for i in range(4):
                ax = fig_draw.add_subplot(2, 2, i+1, projection='3d')
                ax.scatter3D(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2],
                             c=self.Y, alpha=0.7, s=40,
                             cmap='plasma')
                ax.view_init(30 * i, 60 * (i+1))
            fig_draw.tight_layout()
            plt.show()
            fig_draw.savefig("graph/pca.png")

        return X_reduced

    def draw_tSNE(self, perplexity=30,
                  debug=0, n_jobs=1, n_iter=500):
        """Draws training dataset, reduced to 3D projection with t-SNE."""
        tsne = TSNE(n_components=3,  perplexity=perplexity,
                    verbose=debug, n_jobs=n_jobs, n_iter=n_iter,
                    random_state=52)

        X_tsne = tsne.fit_transform(self.draw_PCA(
            reduce_to_95=True,
            draw_expl=False,
            draw_pca=False))

        fig = plt.figure(figsize=(15, 15))
        dimens = self.X.shape[1]
        plt.title(f"{dimens}D data to t-SNE 3D projection")

        for i in range(4):
            ax = fig.add_subplot(2, 2, i+1, projection='3d')
            ax.scatter3D(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2],
                         c=self.Y, alpha=0.7, s=40,
                         cmap='plasma')
            ax.view_init(30 * i, 60 * (i+1))

        fig.tight_layout()
        plt.show()
        fig.savefig("graph/SNE.png")

        return X_tsne
