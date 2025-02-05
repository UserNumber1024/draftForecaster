from math import ceil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_ta as pta
import seaborn as sns
import talib
from lib.connector import MOEXAdapter
import lib.constants as c
from typing import Any
# -----------------------------------------------------------------------------


class MOEXtoXY:
    """Get OHCL-data from MOEX and transform it to train/prediction dataset."""

    def __init__(self, moex_ticker_ids=c.moex_ticker_ids, debug=c.debug):
        """Initialize default data processing and preparation params.

        Parameters
        ----------
        moex_ticker_ids: dict of MOEX tickers for processing (in upcase),
            value is used for converting ticker into a numeric id,
            i.e. into a categorization attribute of the model.
            {'SBER': 0.01}

        """
        self.debug = debug
        if moex_ticker_ids == {}:
            self.moex_ticker_ids = {'SBER': 0.01}
        else:
            self.moex_ticker_ids = moex_ticker_ids
        self.ta_indicators = [
            # 'tickerId',
            'macd12_26_9',
            'macd10_14_5',
            'rsi5',
            'rsi14',
            'rsi8',
            'wpr14',
            'wpr9',
            'wpr21',
            'stoch14_3',
            'stoch9_4',
            'adx14',
            'adx5',
            'dm7',
            'dm12',
            # 'dm21',
            'atr5',
            'atr9',
            'cci14',
            'cci10',
            'stdev11c',
            'stdev11l',
            'stdev11h',
            'stdev6c',
            'stdev6l',
            'stdev6h',
            'bop',
            # 'trix11h',
            # 'trix11l',
            # 'trix11c',
            'trix16h',
            'trix16l',
            # 'trix16c',
            'ema10_ohlc',
            # 'ema25_ohlc',
            # 'wma15_ohlc',
            'wma20_ohlc',
            # 'bb8',
            'bb17',
            'kc21',
            'psar',
            'volume1dInc',
            'volume3dInc',
            'volume5dInc',
            'mfi13',
            'mfi25',
            'ad',
            'adosc12_26',
            'adosc3_10',
            'numTrades1dInc',
            'numTrades3dInc',
            'numTrades5dInc',
            'roc1o',
            'roc3o',
            'roc5o',
            'roc1c',
            'roc3c',
            'roc5c',
            'ibs',
            'close2openInc',
            'high2openInc',
            'low2openInc',
            'dayOfWeek',
            # 'dayOfMonth',
            'dayOfYear',
            # 'numOfMonth',
            'cdl_patterns']
        self.cdl_patterns = [
            # ('CDL2CROWS', talib.CDL2CROWS),
            # ('CDL3BLACKCROWS', talib.CDL3BLACKCROWS),
            ('CDL3INSIDE', talib.CDL3INSIDE),
            # ('CDL3LINESTRIKE', talib.CDL3LINESTRIKE),
            ('CDL3OUTSIDE', talib.CDL3OUTSIDE),
            # ('CDL3STARSINSOUTH', talib.CDL3STARSINSOUTH),
            # ('CDL3WHITESOLDIERS', talib.CDL3WHITESOLDIERS),
            # ('CDLABANDONEDBABY', talib.CDLABANDONEDBABY),
            ('CDLADVANCEBLOCK', talib.CDLADVANCEBLOCK),
            ('CDLBELTHOLD', talib.CDLBELTHOLD),
            # ('CDLBREAKAWAY', talib.CDLBREAKAWAY),
            ('CDLCLOSINGMARUBOZU', talib.CDLCLOSINGMARUBOZU),
            # ('CDLCONCEALBABYSWALL', talib.CDLCONCEALBABYSWALL),
            # ('CDLCOUNTERATTACK', talib.CDLCOUNTERATTACK),
            ('CDLDARKCLOUDCOVER', talib.CDLDARKCLOUDCOVER),
            ('CDLDOJI', talib.CDLDOJI),
            ('CDLDOJISTAR', talib.CDLDOJISTAR),
            ('CDLDRAGONFLYDOJI', talib.CDLDRAGONFLYDOJI),
            ('CDLENGULFING', talib.CDLENGULFING),
            # ('CDLEVENINGDOJISTAR', talib.CDLEVENINGDOJISTAR),
            ('CDLEVENINGSTAR', talib.CDLEVENINGSTAR),
            # ('CDLGAPSIDESIDEWHITE', talib.CDLGAPSIDESIDEWHITE),
            ('CDLGRAVESTONEDOJI', talib.CDLGRAVESTONEDOJI),
            ('CDLHAMMER', talib.CDLHAMMER),
            ('CDLHANGINGMAN', talib.CDLHANGINGMAN),
            ('CDLHARAMI', talib.CDLHARAMI),
            ('CDLHARAMICROSS', talib.CDLHARAMICROSS),
            ('CDLHIGHWAVE', talib.CDLHIGHWAVE),
            ('CDLHIKKAKE', talib.CDLHIKKAKE),
            # ('CDLHIKKAKEMOD', talib.CDLHIKKAKEMOD),
            ('CDLHOMINGPIGEON', talib.CDLHOMINGPIGEON),
            # ('CDLIDENTICAL3CROWS', talib.CDLIDENTICAL3CROWS),
            # ('CDLINNECK', talib.CDLINNECK),
            ('CDLINVERTEDHAMMER', talib.CDLINVERTEDHAMMER),
            # ('CDLKICKING', talib.CDLKICKING),
            # ('CDLKICKINGBYLENGTH', talib.CDLKICKINGBYLENGTH),
            # ('CDLLADDERBOTTOM', talib.CDLLADDERBOTTOM),
            ('CDLLONGLEGGEDDOJI', talib.CDLLONGLEGGEDDOJI),
            ('CDLLONGLINE', talib.CDLLONGLINE),
            ('CDLMARUBOZU', talib.CDLMARUBOZU),
            ('CDLMATCHINGLOW', talib.CDLMATCHINGLOW),
            # ('CDLMATHOLD', talib.CDLMATHOLD),
            # ('CDLMORNINGDOJISTAR', talib.CDLMORNINGDOJISTAR),
            ('CDLMORNINGSTAR', talib.CDLMORNINGSTAR),
            # ('CDLONNECK', talib.CDLONNECK),
            # ('CDLPIERCING', talib.CDLPIERCING),
            ('CDLRICKSHAWMAN', talib.CDLRICKSHAWMAN),
            # ('CDLRISEFALL3METHODS', talib.CDLRISEFALL3METHODS),
            ('CDLSEPARATINGLINES', talib.CDLSEPARATINGLINES),
            ('CDLSHOOTINGSTAR', talib.CDLSHOOTINGSTAR),
            ('CDLSHORTLINE', talib.CDLSHORTLINE),
            ('CDLSPINNINGTOP', talib.CDLSPINNINGTOP),
            ('CDLSTALLEDPATTERN', talib.CDLSTALLEDPATTERN),
            # ('CDLSTICKSANDWICH', talib.CDLSTICKSANDWICH),
            ('CDLTAKURI', talib.CDLTAKURI),
            ('CDLTASUKIGAP', talib.CDLTASUKIGAP),
            ('CDLTHRUSTING', talib.CDLTHRUSTING),
            # ('CDLTRISTAR', talib.CDLTRISTAR),
            # ('CDLUNIQUE3RIVER', talib.CDLUNIQUE3RIVER),
            # ('CDLUPSIDEGAP2CROWS', talib.CDLUPSIDEGAP2CROWS),
            ('CDLXSIDEGAP3METHODS', talib.CDLXSIDEGAP3METHODS)]

        self.X = None
        self.Y = None

    def __log(self, text: Any, divider=True):
        """Print log info if 'debug' is on."""
        if self.debug >= 1:
            if divider:
                print('{:-^50}'.format("-"))
            print('-=[ ', text, ' ]=-')

    def prepare_XY(self,
                   start=c.start,
                   end=c.end,
                   store_tickers2file=True,
                   store_XY2file=True,
                   file_folder=c.file_folder,
                   source_tickers=c.file,
                   source_XY=c.calc,
                   profit_margin=c.profit_margin,
                   n_days=c.n_days,
                   up_quant=c.up_quant,
                   low_quant=c.low_quant,
                   drop_head_tail=True
                   ):
        """Prepare X and Y for analysis and forecasting.

        The source data can be extracted from files (from previos run`s)
        or obtained from MOEX (with the possibility of saving to a file).

        For forecasting, request data for at least 3 month
        due to specific trading indicators calculation.
        (while forecasting, set store_tickers2file=False)

        X and Y keeped as class attributes (it's expensive, but whatever)

        Parameters
        ----------
        start, end: str, period in "YYYY-MM-DD" format
            default 2015..2023 (not used when source_tickers=file)

        store_tickers2file: Boolean, default True
            define, whether the "raw" trading data should be saved to a file

        store_XY2file:Boolean, default True
            define, whether prepared X and Y should be saved to a file

        source_tickers: {'file', 'moex'}, default 'file', defines the method
        of obtaining trade data
            * from the 'moex' (MOEX API)
            * or from a 'file' (from a previous run)

        source_XY: {'file', 'calc'}, default 'calc', defines the method
        of preparing X and Y
            * 'calc' calculate due params
            * 'file' read from previos preparings

        profit_margin: float, default 0.01
            is used to calculate Y (profit), determines price increase
            (in percent) at which potential profit is calculated

        n_days: days for normalization by mean
            default 60

        up_quant, low_quant:  used to normalize/scale indicators
            default 0.95, 0.05

        drop_head_tail: set False while forecasting, True for model training.

        Return
        ------
        X and Y to use by models.
        Y consist from columns `profit_x` and `mean_delta_x`, where x in [1..5]
          `profit` - will there be a profit in x days to the current price
          `mean_delta` - shows the price change after x days
        Also return prices - OHLCV from MOEX enriched with X and Y.

        """
        self.__drop_head_tail = drop_head_tail
        self.__log('ticker source: ' + source_tickers)

        if source_XY == c.file:
            X = pd.read_csv(file_folder + 'X.csv', index_col=0)
            self.__log('load X from: ' + file_folder, False)

            Y = pd.read_csv(file_folder + 'Y.csv', index_col=0)
            self.__log('load Y from: ' + file_folder, False)

            self.X = X
            self.Y = Y

            prices = pd.read_csv(file_folder + 'prices.csv', index_col=0)
            self.__log('load prices from: ' + file_folder, False)

            return X, Y, prices

        tickers = list(self.moex_ticker_ids.keys())
        X = pd.DataFrame()
        Y = pd.DataFrame()
        prices = pd.DataFrame()
        self.profit_margin = profit_margin
        self.n_days = n_days
        self.up_quant = up_quant
        self.low_quant = low_quant

        self.__log('Profit is ' + str(self.profit_margin) + ' margin', False)

        if source_tickers == c.moex:
            iss = MOEXAdapter()

        for ticker in tickers:
            filename = file_folder + ticker + '.csv'
            if source_tickers == c.moex:
                ticker_data = iss.get_ticker_history(
                    ticker, start, end).reset_index(drop=True)
                self.__log(ticker + ' load from moex', False)
                if store_tickers2file:
                    ticker_data.to_csv(filename)
                    self.__log('tickers saved to: ' + filename, False)
            elif source_tickers == c.file:
                ticker_data = pd.read_csv(filename)
                self.__log(ticker + ' load from ' + filename, False)

            x_ticker, y_ticker, prices_tck = self.__calc_TA_profit(ticker_data)
            X = pd.concat([X, x_ticker])
            Y = pd.concat([Y, y_ticker])
            prices = pd.concat([prices, prices_tck])
            X.reset_index(drop=True, inplace=True)
            Y.reset_index(drop=True, inplace=True)
            prices.reset_index(drop=True, inplace=True)

        self.X = X
        self.Y = Y

        if store_XY2file:
            X.to_csv(file_folder + 'X.csv')
            self.__log('save X to: ' + file_folder, False)

            Y.to_csv(file_folder + 'Y.csv')
            self.__log('save Y to: ' + file_folder, False)

            prices.to_csv(file_folder + 'prices.csv')
            self.__log('save prices to: ' + file_folder, False)
        return X, Y, prices

    def __get_OHLCV(self, moex_data):
        """Extract OHLCV columns from MOEX full data, rename to common ones."""
        prices = moex_data[["TRADEDATE", "OPEN", "LOW",
                            "HIGH", "LEGALCLOSEPRICE",
                            "VOLUME", "NUMTRADES", "SECID"]]
        prices.columns = ['Date', 'Open', 'Low',
                          'High', 'Close', 'Volume', 'Numtrades', 'Ticker']
        self.__log('OHLCV extracted from MOEX data')
        return prices

    def __calc_TA_profit(self, moex_ticker_data: pd.DataFrame):
        """Generate indicators for model training and forecasting.

        Parameters
        ----------
        moex_ticker_data
            MOEX API-response in pd.DataFrame form,

        Return
        ------
        for specific ticker:
            - X and Y to use by models.
            - prices - OHLCV from MOEX enriched with X and Y.

        Useful links
        ------------
        https://tradingstrategy.ai/docs/api/technical-analysis/index.html
        https://ta-lib.github.io/ta-lib-python/doc_index.html
        """
        prices = self.__get_OHLCV(moex_ticker_data)

        # remove rows with OPEN=0
        # cause it looks like there was no trading that day
        prices = prices[prices.Open > 0]
        prices = prices[prices.Volume > 0]

        X = pd.DataFrame()
        Y = pd.DataFrame()

        # Split HLOС into pd.Series just for ease of use
        H = prices['High']
        L = prices['Low']
        Op = prices['Open']
        C = prices['Close']
        V = prices['Volume']
        N = prices['Numtrades']
        Date = pd.to_datetime(prices['Date'])

        n_days = self.n_days
        up_quant = self.up_quant
        low_quant = self.low_quant

        for i in self.ta_indicators:
            if i == 'tickerId':
                # convert ticker into a numeric id,
                # i.e. into a categorization attribute of the model
                T = prices['Ticker']
                T = T.replace(self.moex_ticker_ids)
                T.name = 'TickerId'
                X = pd.concat([X, pd.DataFrame(T)])

            # MACD normalized by the n-day mean price (HLC)
            elif i == 'macd12_26_9':
                macd = pta.macd(close=C, fast=12, slow=26, signal=9)
                mean = pta.hlc3(H, L, C)
                macdm = macd.iloc[:, 0]
                macdm = macdm / mean.rolling(window=n_days).mean()
                macdh = macd.iloc[:, 1]
                macdh = macdh / mean.rolling(window=n_days).mean()
                X = pd.concat([X, pd.Series(macdm, name=i),
                               pd.Series(macdh, name=i + 'hist'), ], axis=1)
                # macds = macd.iloc[:, 2]
                # macds = macds / mean.rolling(window=n_days).mean()
                # X = pd.concat([X, pd.Series(macdm, name=i),
                #                pd.Series(macdh, name=i + 'hist'),
                #                pd.Series(macds, name=i + 'sig')], axis=1)
            elif i == 'macd10_14_5':
                macd = pta.macd(close=C, fast=10, slow=14, signal=5)
                mean = pta.hlc3(H, L, C)
                macdm = macd.iloc[:, 0]
                macdm = macdm / mean.rolling(window=n_days).mean()
                macdh = macd.iloc[:, 1]
                macdh = macdh / mean.rolling(window=n_days).mean()
                X = pd.concat([X, pd.Series(macdm, name=i),
                               pd.Series(macdh, name=i + 'hist')], axis=1)
                # macds = macd.iloc[:, 2]
                # macds = macds / mean.rolling(window=n_days).mean()
                # X = pd.concat([X, pd.Series(macdm, name=i),
                #                pd.Series(macdh, name=i + 'hist'),
                #                pd.Series(macds, name=i + 'sig')], axis=1)

            elif i == 'rsi5':
                X = pd.concat([X, pta.rsi(close=C, length=5) / 100], axis=1)
            elif i == 'rsi14':
                X = pd.concat([X, pta.rsi(close=C, length=14) / 100], axis=1)
            elif i == 'rsi8':
                X = pd.concat([X, pta.rsi(close=C, length=8) / 100], axis=1)

            elif i == 'wpr9':
                wpr = pta.willr(high=H, low=L, close=C, length=9) / 100 * -1
                X = pd.concat([X, wpr], axis=1)
            elif i == 'wpr14':
                wpr = pta.willr(high=H, low=L, close=C, length=14) / 100 * -1
                X = pd.concat([X, wpr], axis=1)
            elif i == 'wpr21':
                wpr = pta.willr(high=H, low=L, close=C, length=21) / 100 * -1
                X = pd.concat([X, wpr], axis=1)

            elif i == 'stoch14_3':
                stoch = pta.stoch(high=H, low=L, close=C, k=14, d=3) / 100
                X = pd.concat([X, stoch], axis=1)
            elif i == 'stoch9_4':
                stoch = pta.stoch(high=H, low=L, close=C, k=9, d=4) / 100
                X = pd.concat([X, stoch], axis=1)

            elif i == 'adx5':
                adx = pta.adx(high=H, low=L, close=C, length=5) / 100
                X = pd.concat([X, adx], axis=1)
            elif i == 'adx14':
                adx = pta.adx(high=H, low=L, close=C, length=14) / 100
                X = pd.concat([X, adx], axis=1)

            # Directional Movement normalized by the n-day DM
            elif i == 'dm7':
                dm = pta.dm(high=H, low=L, length=7)
                dmplus = dm.iloc[:, 0] / 100
                dmplus = dmplus / dmplus.rolling(window=n_days).mean()
                dmplus = dmplus.clip(upper=dmplus.quantile(up_quant)) / 2
                dmminus = dm.iloc[:, 1] / 100
                dmminus = dmminus / dmminus.rolling(window=n_days).mean()
                dmminus = dmminus.clip(upper=dmminus.quantile(up_quant)) / 2
                X = pd.concat([X, pd.Series(dmplus, name=i + '+'),
                               pd.Series(dmminus, name=i + '-')], axis=1)
            elif i == 'dm12':
                dm = pta.dm(high=H, low=L, length=12)
                dmplus = dm.iloc[:, 0] / 100
                dmplus = dmplus / dmplus.rolling(window=n_days).mean()
                dmplus = dmplus.clip(upper=dmplus.quantile(up_quant)) / 2
                dmminus = dm.iloc[:, 1] / 100
                dmminus = dmminus / dmminus.rolling(window=n_days).mean()
                dmminus = dmminus.clip(upper=dmminus.quantile(up_quant)) / 2
                X = pd.concat([X, pd.Series(dmplus, name=i + '+'),
                               pd.Series(dmminus, name=i + '-')], axis=1)
            elif i == 'dm21':
                dm = pta.dm(high=H, low=L, length=21)
                dmplus = dm.iloc[:, 0] / 100
                dmplus = dmplus / dmplus.rolling(window=n_days).mean()
                dmplus = dmplus.clip(upper=dmplus.quantile(up_quant)) / 2
                dmminus = dm.iloc[:, 1] / 100
                dmminus = dmminus / dmminus.rolling(window=n_days).mean()
                dmminus = dmminus.clip(upper=dmminus.quantile(up_quant)) / 2
                X = pd.concat([X, pd.Series(dmplus, name=i + '+'),
                               pd.Series(dmminus, name=i + '-')], axis=1)

            # Average True Range normalized by the n-day mean price (HLC)
            elif i == 'atr5':
                atr = pta.atr(high=H, low=L, close=C, length=5)
                atr = atr / atr.rolling(window=n_days).mean()
                atr = atr.clip(upper=atr.quantile(up_quant))
                X = pd.concat([X, pd.Series(atr, name=i)], axis=1)
            elif i == 'atr9':
                atr = pta.atr(high=H, low=L, close=C, length=9)
                atr = atr / atr.rolling(window=n_days).mean()
                atr = atr.clip(upper=atr.quantile(up_quant))
                X = pd.concat([X, pd.Series(atr, name=i)], axis=1)

            elif i == 'cci10':
                cci = pta.cci(high=H, low=L, close=C, length=10) / 100 / 5
                X = pd.concat([X, cci], axis=1)
            elif i == 'cci14':
                cci = pta.cci(high=H, low=L, close=C, length=14) / 100 / 5
                X = pd.concat([X, cci], axis=1)

            # STDEV normalized by the n-day STDEV
            elif i == 'stdev11c':
                stdev = pta.stdev(close=C, length=11)
                stdev = stdev / stdev.rolling(window=n_days).mean() / 2
                stdev = stdev.clip(upper=stdev.quantile(up_quant))
                X = pd.concat([X, pd.Series(stdev, name=i)], axis=1)
            elif i == 'stdev11l':
                stdev = pta.stdev(close=L, length=11)
                stdev = stdev / stdev.rolling(window=n_days).mean() / 2
                stdev = stdev.clip(upper=stdev.quantile(up_quant))
                X = pd.concat([X, pd.Series(stdev, name=i)], axis=1)
            elif i == 'stdev11h':
                stdev = pta.stdev(close=H, length=11)
                stdev = stdev / stdev.rolling(window=n_days).mean() / 2
                stdev = stdev.clip(upper=stdev.quantile(up_quant))
                X = pd.concat([X, pd.Series(stdev, name=i)], axis=1)
            elif i == 'stdev6c':
                stdev = pta.stdev(close=C, length=6)
                stdev = stdev / stdev.rolling(window=n_days).mean() / 2
                stdev = stdev.clip(upper=stdev.quantile(up_quant))
                X = pd.concat([X, pd.Series(stdev, name=i)], axis=1)
            elif i == 'stdev6l':
                stdev = pta.stdev(close=L, length=6)
                stdev = stdev / stdev.rolling(window=n_days).mean() / 2
                stdev = stdev.clip(upper=stdev.quantile(up_quant))
                X = pd.concat([X, pd.Series(stdev, name=i)], axis=1)
            elif i == 'stdev6h':
                stdev = pta.stdev(close=H, length=6)
                stdev = stdev / stdev.rolling(window=n_days).mean() / 2
                stdev = stdev.clip(upper=stdev.quantile(up_quant))
                X = pd.concat([X, pd.Series(stdev, name=i)], axis=1)

            elif i == 'bop':
                X = pd.concat(
                    [X, pta.bop(open_=Op, high=H, low=L, close=C)], axis=1)

            elif i == 'trix11c':
                trix = pta.trix(close=C, length=11)
                trixsig = trix.iloc[:, 1]
                trixsig = trixsig.clip(lower=trixsig.quantile(low_quant),
                                       upper=trixsig.quantile(up_quant))
                trix = trix.iloc[:, 0]
                trix = trix.clip(lower=trix.quantile(low_quant),
                                 upper=trix.quantile(up_quant))
                trix.columns = [i, i + 's']
                X = pd.concat([X, pd.Series(trix, name=i),
                               pd.Series(trixsig, name=i + 'sig')], axis=1)
            elif i == 'trix11l':
                trix = pta.trix(close=L, length=11)
                trixsig = trix.iloc[:, 1]
                trixsig = trixsig.clip(lower=trixsig.quantile(low_quant),
                                       upper=trixsig.quantile(up_quant))
                trix = trix.iloc[:, 0]
                trix = trix.clip(lower=trix.quantile(low_quant),
                                 upper=trix.quantile(up_quant))
                trix.columns = [i, i + 's']
                X = pd.concat([X, pd.Series(trix, name=i),
                               pd.Series(trixsig, name=i + 'sig')], axis=1)
            elif i == 'trix11h':
                trix = pta.trix(close=H, length=11)
                trixsig = trix.iloc[:, 1]
                trixsig = trixsig.clip(lower=trixsig.quantile(low_quant),
                                       upper=trixsig.quantile(up_quant))
                trix = trix.iloc[:, 0]
                trix = trix.clip(lower=trix.quantile(low_quant),
                                 upper=trix.quantile(up_quant))
                trix.columns = [i, i + 's']
                X = pd.concat([X, pd.Series(trix, name=i),
                               pd.Series(trixsig, name=i + 'sig')], axis=1)

            elif i == 'trix16c':
                trix = pta.trix(close=C, length=16)
                trixsig = trix.iloc[:, 1]
                trixsig = trixsig.clip(lower=trixsig.quantile(low_quant),
                                       upper=trixsig.quantile(up_quant))
                trix = trix.iloc[:, 0]
                trix = trix.clip(lower=trix.quantile(low_quant),
                                 upper=trix.quantile(up_quant))
                trix.columns = [i, i + 's']
                X = pd.concat([X, pd.Series(trix, name=i),
                               pd.Series(trixsig, name=i + 'sig')], axis=1)
            elif i == 'trix16l':
                trix = pta.trix(close=L, length=16)
                trixsig = trix.iloc[:, 1]
                trixsig = trixsig.clip(lower=trixsig.quantile(low_quant),
                                       upper=trixsig.quantile(up_quant))
                trix = trix.iloc[:, 0]
                trix = trix.clip(lower=trix.quantile(low_quant),
                                 upper=trix.quantile(up_quant))
                trix.columns = [i, i + 's']
                X = pd.concat([X, pd.Series(trix, name=i),
                               pd.Series(trixsig, name=i + 'sig')], axis=1)
            elif i == 'trix16h':
                trix = pta.trix(close=H, length=16)
                trixsig = trix.iloc[:, 1]
                trixsig = trixsig.clip(lower=trixsig.quantile(low_quant),
                                       upper=trixsig.quantile(up_quant))
                trix = trix.iloc[:, 0]
                trix = trix.clip(lower=trix.quantile(low_quant),
                                 upper=trix.quantile(up_quant))
                trix.columns = [i, i + 's']
                X = pd.concat([X, pd.Series(trix, name=i),
                               pd.Series(trixsig, name=i + 'sig')], axis=1)

            # EMA(mean price) normalized by the n-day mean price (HLC)
            elif i == 'ema10_ohlc':
                ema = pta.ema(pta.ohlc4(Op, H, L, C), length=10)
                mean = pta.hlc3(H, L, C)
                ema = ema / mean.rolling(window=n_days).mean()
                X = pd.concat([X, pd.Series(ema, name=i)], axis=1)
            elif i == 'ema25_ohlc':
                ema = pta.ema(pta.ohlc4(Op, H, L, C), length=25)
                mean = pta.hlc3(H, L, C)
                ema = ema / mean.rolling(window=n_days).mean()
                X = pd.concat([X, pd.Series(ema, name=i)], axis=1)

            # WMA (mean price) normalized by the n-day mean price (HLC)
            elif i == 'wma15_ohlc':
                wma = pta.wma(pta.ohlc4(Op, H, L, C), length=15)
                mean = pta.hlc3(H, L, C)
                wma = wma / mean.rolling(window=n_days).mean()
                X = pd.concat([X, pd.Series(wma, name=i)], axis=1)
            elif i == 'wma20_ohlc':
                wma = pta.wma(pta.ohlc4(Op, H, L, C), length=20)
                mean = pta.hlc3(H, L, C)
                wma = wma / mean.rolling(window=n_days).mean()
                X = pd.concat([X, pd.Series(wma, name=i)], axis=1)

            # Bollinger Bands normalized by the n-day mean price (HLC)
            elif i == 'bb8':
                bb = pta.bbands(C, length=8)
                mean = pta.hlc3(H, L, C)
                low = bb.iloc[:, 0] / mean.rolling(window=n_days).mean()
                mid = bb.iloc[:, 1] / mean.rolling(window=n_days).mean()
                hi = bb.iloc[:, 2] / mean.rolling(window=n_days).mean()
                bw = bb.iloc[:, 3] / 100
                bw = bw.clip(lower=bw.quantile(low_quant),
                             upper=bw.quantile(up_quant))
                perc = bb.iloc[:, 4]
                perc = perc.clip(lower=perc.quantile(low_quant),
                                 upper=perc.quantile(up_quant))
                X = pd.concat([X, pd.Series(low, name=i + 'low'),
                               pd.Series(mid, name=i + 'mid'),
                               pd.Series(hi, name=i + 'hi'),
                               pd.Series(bw, name=i + 'bw'),
                               pd.Series(perc, name=i + 'perc')], axis=1)
            elif i == 'bb17':
                bb = pta.bbands(C, length=17)
                mean = pta.hlc3(H, L, C)
                low = bb.iloc[:, 0] / mean.rolling(window=n_days).mean()
                mid = bb.iloc[:, 1] / mean.rolling(window=n_days).mean()
                hi = bb.iloc[:, 2] / mean.rolling(window=n_days).mean()
                bw = bb.iloc[:, 3] / 100
                bw = bw.clip(lower=bw.quantile(low_quant),
                             upper=bw.quantile(up_quant))
                perc = bb.iloc[:, 4]
                perc = perc.clip(lower=perc.quantile(low_quant),
                                 upper=perc.quantile(up_quant))
                X = pd.concat([X, pd.Series(low, name=i + 'low'),
                               pd.Series(mid, name=i + 'mid'),
                               pd.Series(hi, name=i + 'hi'),
                               pd.Series(bw, name=i + 'bw'),
                               pd.Series(perc, name=i + 'perc')], axis=1)

            # Keltner Channels normalized by the n-day mean price (HLC)
            elif i == 'kc21':
                kc = pta.kc(high=H, low=L, close=C, length=21)
                mean = pta.hlc3(H, L, C)
                low = kc.iloc[:, 0] / mean.rolling(window=n_days).mean()
                bas = kc.iloc[:, 1] / mean.rolling(window=n_days).mean()
                up = kc.iloc[:, 2] / mean.rolling(window=n_days).mean()
                X = pd.concat([X, pd.Series(low, name=i + 'low'),
                               pd.Series(bas, name=i + 'bas'),
                               pd.Series(up, name=i + 'up')], axis=1)

            # Parabolic Stop/Reverse normalized by the n-day mean price (HLC)
            elif i == 'psar':
                psar = pta.psar(high=H, low=L, close=C, fillna=0)
                mean = pta.hlc3(H, L, C)
                long = psar.iloc[:, 0] / mean.rolling(window=n_days).mean()
                short = psar.iloc[:, 1] / mean.rolling(window=n_days).mean()
                af = psar.iloc[:, 2]
                rev = psar.iloc[:, 3]
                X = pd.concat([X, pd.Series(long, name=i + 'long'),
                               pd.Series(short, name=i + 'short'),
                               pd.Series(af, name=i + 'af'),
                               pd.Series(rev, name=i + 'rev')], axis=1)

            #  divided by 10 because "volume" is very volatile.
            elif i == 'volume1dInc':
                vol = (V - V.shift(1)) / V.shift(1) / 10
                vol = vol.clip(upper=vol.quantile(up_quant))
                X = pd.concat([X, pd.Series(vol, name=i)], axis=1)

            elif i == 'volume3dInc':
                vol = (V - V.shift(3)) / V.shift(3) / 10
                vol = vol.clip(upper=vol.quantile(up_quant))
                X = pd.concat([X, pd.Series(vol, name=i)], axis=1)

            elif i == 'volume5dInc':
                vol = (V - V.shift(5)) / V.shift(5) / 10
                vol = vol.clip(upper=vol.quantile(up_quant))
                X = pd.concat([X, pd.Series(vol, name=i)], axis=1)

            elif i == 'mfi13':
                mf = pta.mfi(high=H, low=L, close=C, volume=V, length=13) / 100
                X = pd.concat([X, mf], axis=1)
            elif i == 'mfi25':
                mf = pta.mfi(high=H, low=L, close=C, volume=V, length=25) / 100
                X = pd.concat([X, mf], axis=1)

            # AD normalized by the n-day mean volume
            elif i == 'ad':
                ad = pta.ad(high=H, low=L, close=C, volume=V, open_=Op)
                ad = ad / (V.rolling(window=n_days).mean() * 100) / 2
                ad = ad.clip(lower=ad.quantile(low_quant),
                             upper=ad.quantile(up_quant))
                X = pd.concat([X, pd.Series(ad, name=i)], axis=1)

            # ADOSC normalized by the n-day mean volume
            elif i == 'adosc12_26':
                adosc = pta.adosc(high=H, low=L, close=C,
                                  volume=V, open_=Op, fast=12, slow=26)
                adosc = adosc / V.rolling(window=n_days).mean() / 2
                adosc = adosc.clip(lower=adosc.quantile(low_quant),
                                   upper=adosc.quantile(up_quant))
                X = pd.concat([X, pd.Series(adosc, name=i)], axis=1)
            elif i == 'adosc3_10':
                adosc = pta.adosc(high=H, low=L, close=C,
                                  volume=V, open_=Op, fast=3, slow=10)
                adosc = adosc / V.rolling(window=n_days).mean() / 2
                adosc = adosc.clip(lower=adosc.quantile(low_quant),
                                   upper=adosc.quantile(up_quant))
                X = pd.concat([X, pd.Series(adosc, name=i)], axis=1)

            # divided by 10 because "number of trades" is very volatile.
            elif i == 'numTrades1dInc':
                numt = (N - N.shift(1)) / N.shift(1) / 10
                numt = numt.clip(upper=numt.quantile(up_quant))
                X = pd.concat([X, pd.Series(numt, name=i)], axis=1)
            elif i == 'numTrades3dInc':
                numt = (N - N.shift(3)) / N.shift(3) / 10
                numt = numt.clip(upper=numt.quantile(up_quant))
                X = pd.concat([X, pd.Series(numt, name=i)], axis=1)
            elif i == 'numTrades5dInc':
                numt = (N - N.shift(5)) / N.shift(5) / 10
                numt = numt.clip(upper=numt.quantile(up_quant))
                X = pd.concat([X, pd.Series(numt, name=i)], axis=1)

            # rate of change
            elif i == 'roc1o':
                roc = (Op - Op.shift(1)) / Op.shift(1)
                roc = roc.clip(lower=roc.quantile(low_quant),
                               upper=roc.quantile(up_quant))
                X = pd.concat([X, pd.Series(roc, name=i)], axis=1)
            elif i == 'roc3o':
                roc = (Op - Op.shift(3)) / Op.shift(3)
                roc = roc.clip(lower=roc.quantile(low_quant),
                               upper=roc.quantile(up_quant))
                X = pd.concat([X, pd.Series(roc, name=i)], axis=1)
            elif i == 'roc5o':
                roc = (Op - Op.shift(5)) / Op.shift(5)
                roc = roc.clip(lower=roc.quantile(low_quant),
                               upper=roc.quantile(up_quant))
                X = pd.concat([X, pd.Series(roc, name=i)], axis=1)

            elif i == 'roc1c':
                roc = (C - C.shift(1)) / C.shift(1)
                roc = roc.clip(lower=roc.quantile(low_quant),
                               upper=roc.quantile(up_quant))
                X = pd.concat([X, pd.Series(roc, name=i)], axis=1)
            elif i == 'roc3c':
                roc = (C - C.shift(3)) / C.shift(3)
                roc = roc.clip(lower=roc.quantile(low_quant),
                               upper=roc.quantile(up_quant))
                X = pd.concat([X, pd.Series(roc, name=i)], axis=1)
            elif i == 'roc5c':
                roc = (C - C.shift(5)) / C.shift(5)
                roc = roc.clip(lower=roc.quantile(low_quant),
                               upper=roc.quantile(up_quant))
                X = pd.concat([X, pd.Series(roc, name=i)], axis=1)

            # Internal Bar Strength
            elif i == 'ibs':
                X = pd.concat(
                    [X, pd.Series((C - L) / (H - L), name=i)], axis=1)

            elif i == 'close2openInc':
                X = pd.concat([X, pd.Series((C - Op) / Op, name=i)], axis=1)

            elif i == 'high2openInc':
                X = pd.concat([X, pd.Series((H - Op) / Op, name=i)], axis=1)

            elif i == 'low2openInc':
                X = pd.concat([X, pd.Series((L - Op) / Op, name=i)], axis=1)

            elif i == 'dayOfWeek':
                X = pd.concat(
                    [X, pd.Series(Date.dt.dayofweek / 7, name='DoW')], axis=1)
            elif i == 'dayOfMonth':
                X = pd.concat(
                    [X, pd.Series(Date.dt.day / 31, name='DoMnth')], axis=1)
            elif i == 'dayOfYear':
                X = pd.concat(
                    [X, pd.Series(Date.dt.dayofyear / 365, name='DoYear')],
                    axis=1)
            elif i == 'numOfMonth':
                X = pd.concat(
                    [X, pd.Series(Date.dt.month / 12, name='Month')], axis=1)
            elif i == 'cdl_patterns':
                for patt, fnc in self.cdl_patterns:
                    cdl = pd.Series(fnc(Op, H, L, C)/100, name='cdl_' + patt)
                    X = pd.concat([X, cdl], axis=1)

        # Y - profit
        # if buy now, will be there an average profit after N days?
        # assume, that for classification, the profit should be
        # at least 1 percent from transaction
        # calculated by mean price increment for N days
        mean = pta.hlc3(H, L, C)
        for per in [1, 2, 3, 4, 5]:
            mean_forecast = pta.hlc3(H.shift(-per), L.shift(-per),
                                     C.shift(-per))
            delta = (mean_forecast - mean) / mean
            Y = pd.concat([Y,
                           pd.Series((delta >= self.profit_margin),
                                     name='profit_'+str(per)).astype(int)],
                          axis=1)
            Y = pd.concat(
                [Y, pd.Series((delta), name='mean_delta_'+str(per))], axis=1)

        X = X.round(8)
        Y = Y.round(8)
        mean = mean.round(8)

        prices = pd.concat([prices, X, Y, mean],
                           axis=1).reindex(prices.index)

        if self.__drop_head_tail:
            # drop lastN rows, because Y not valid there
            X.drop(X.tail(5).index, inplace=True)
            Y.drop(Y.tail(5).index, inplace=True)
            prices.drop(prices.tail(5).index, inplace=True)

            # drop n_days first records
            # X not valid here due n_days normalization
            X.drop(X.head(n_days+26).index, inplace=True)
            Y.drop(Y.head(n_days+26).index, inplace=True)
            prices.drop(prices.head(n_days+26).index, inplace=True)
            self.__log('head / tail dropped', False)

        prices = prices.fillna(-1000).replace(np.inf, -1000)
        X = X.fillna(-1000).replace(np.inf, -1000)
        Y = Y.fillna(-1000).replace(np.inf, -1000)

        self.__log('X&Y calculated', False)
        return X, Y, prices

    def draw_X_Y(self, graph_folder=c.graph_folder):
        """Visualisation of training dataset."""
        h = ceil(len(self.X.columns) ** (1 / 2))
        fig_hist = plt.figure(figsize=(30, 30))
        for i, column in enumerate(self.X.columns):
            plt.subplot(h, h, i + 1)
            sns.histplot(data=self.X[column], kde=True)
            plt.axvline(self.X[column].mean(), color='green', linestyle='--')
            plt.axvline(self.X[column].median(), color='black', linestyle='--')

        plt.tight_layout()
        fig_hist.savefig(graph_folder + "x_hist.png")
        self.__log('save x_hist to: ' + graph_folder)

        fig_box = plt.figure(figsize=(150, 150))
        sns.boxplot(data=self.X, palette="Set1", showmeans=True, orient='w')
        plt.grid(True)
        plt.xticks(rotation='vertical')
        fig_box.savefig(graph_folder + "x_box.png")
        self.__log('save x_box to: ' + graph_folder, False)

        fig_heat = plt.figure(figsize=(150, 150), dpi=80)
        sns.heatmap(self.X.corr(), cmap='RdYlGn', annot=False, linewidths=1,
                    linecolor='white')
        fig_heat.savefig(graph_folder + "x_heatmap.png")
        self.__log('save x_heatmap to: ' + graph_folder, False)

        correlation_matrix = self.X.corr()
        correlation_matrix = correlation_matrix.where(
            ~np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        groups = []
        for i in range(len(correlation_matrix.columns)):
            group = set()
            for j in range(len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i, j]) > 0.95 and i != j:
                    group.add(correlation_matrix.columns[i])
                    group.add(correlation_matrix.columns[j])
            if group and group not in groups:
                groups.append(group)
        with open(graph_folder + "x_corr.txt", 'w', encoding='utf-8') as file:
            for idx, group in enumerate(groups):
                file.write(f"High_Corr {idx + 1}: {group}"+'\n')
        self.__log('save x_corr to: ' + graph_folder, False)

        # fig_Y = plt.figure(figsize=(10, 10))
        # sns.histplot(data=self.Y['profit'])
        # fig_Y.savefig(graph_folder + "y_hist.png")
        # self.__log('save y_hist to: ' + graph_folder, False)
