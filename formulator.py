import pandas as pd
import pandas_ta as pta
import numpy as np


class OHLCV2MLdata:
    """Transform ticker-OHLCV to prepared for for train/prediction dataset.

    Attributes
    ----------
    moex_ticker_ids : dict of supported tickers
        used for converting ticker into a numeric id;
        i.e. into a categorization attribute of the model

    moex_indicators : list of supported indicators
        if necessary must be changed before starting the calculation.

    Methods
    -------
    сalc_TA_profit

    See Also
    --------
    useful link: https://tradingstrategy.ai/docs/api/index.html
    """

    def __init__(self):
        self.__ticker: str
        self.moex_ticker_ids = {'SBER': 0.1,
                                'GAZP': 0.2,
                                'LKOH': 0.3
                                }
        self.moex_indicators = [
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
            'dayOfWeek'
        ]

    def __get_OHLCV(self, moex_data):
        """Extract OHLCV columns from MOEX full data, rename to common ones."""
        prices = moex_data[["TRADEDATE", "OPEN", "LOW",
                            "HIGH", "LEGALCLOSEPRICE",
                           "VOLUME", "NUMTRADES", "SECID"]]
        prices.columns = ['Date', 'Open', 'Low',
                          'High', 'Close', 'Volume', 'Numtrades', 'Ticker']
        return prices

    def сalc_TA_profit(self, ticker: str, moex_data: pd.DataFrame,
                       length=1, Y_type='clf'):
        """Generate indicators for model training and forecasting.

        ticker - MOEX ticker in upcase, e.g. "SBER" (see getMoex_ticker_ids),
        moex_data - MOEX API-response in pd.DataFrame form,
        length - on which day calc profit.
        Y_type - 'clf' or 'regr' Y to generate

        Return X and Y for use by models.
        Also return prices - OHLCV from MOEX enriched with X and Y.
        """
        self.__ticker = ticker
        prices = self.__get_OHLCV(moex_data)

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

        # convert ticker into a numeric id,
        # i.e. into a categorization attribute of the model
        T = prices['Ticker']
        T = T.replace(self.moex_ticker_ids)
        T.name = 'TickerId'
        X = pd.concat([X, pd.DataFrame(T)])

        indicators = self.moex_indicators
        for i in indicators:
            if i == 'macd12_26_9':
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
                # ^^^^^^возможно есть смысл сделать по всем столбцам

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
            # ^^^^^^возможно есть смысл сделать по всем столбцам

            elif i == 'stdev15':
                X = pd.concat([X, pta.stdev(close=C, length=15)],
                              axis=1).fillna(0).replace(np.inf, 0)
            # ^^^^^^возможно есть смысл сделать по всем столбцам

            elif i == 'bop':
                X = pd.concat([X, pta.bop(open_=Op, high=H, low=L, close=C)],
                              axis=1).fillna(0).replace(np.inf, 0)
            elif i == 'trix14':
                X = pd.concat([X, pta.trix(close=C, length=14)],
                              axis=1).fillna(0).replace(np.inf, 0)
            # ^^^^^^возможно есть смысл сделать по всем столбцам

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

        if Y_type == 'clf':
            Y = pd.Series((profit >= 0.01), name='profit').astype(int)
        elif Y_type == 'regr':
            Y = pd.Series(profit, name='profit')

        prices = pd.concat([prices, X, profit, Y, mean],
                           axis=1).reindex(
            prices.index).fillna(-1000).replace(np.inf, -1000)

        X.drop(X.tail(length).index, inplace=True)
        Y.drop(Y.tail(length).index, inplace=True)
        prices.drop(prices.tail(length).index, inplace=True)

        return X, Y, prices
