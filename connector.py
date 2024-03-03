import requests
import pandas as pd

# some default api-params:
#   off = turn off `metadata` object in api response
#   extended = generate 'normal' json objects
default_params = {"iss.meta": "off",
                  "iss.json": "extended"}


class MOEXAdapter:
    """Connector for getting data from MOEX ISS.

    Loads data using GET http-requests to the appropriate REST API

    Supports automatic loading of multipage responses

    Methods
    -------
    getTickerHistory

    get_ticker_candles

    See Also
    --------
        full MOEX API`s list:
            https://iss.moex.com/iss/reference/
        developer manual:
            https://fs.moex.com/files/6523
        columns description:
            https://www.moex.com/s1194
    """

    def __init__(self):
        self.__session = requests.Session

        # api-params (http-request params)
        self.__params = {}
        self.__endpoint = None

        # name of object, expected in api response, e.g. 'history' or 'candles'
        self.__parsing_objest = None

    def __prepare_params(self,
                         from_: str = None, till: str = None,
                         columns: str = None, interval: int = None):
        """Add params to http-request (if given by the call).

        e.g. period or specific columns
        """
        if from_:
            self.__params["from"] = from_
        if till:
            self.__params["till"] = till
        if columns:
            self.__params[self.__parsing_objest+".columns"] = columns
        if interval:
            self.__params["interval"] = interval

    def __get_one_page(self, start: int = None):
        """Perform request to get one page.

        (MOEX_API returns paginated data)
        """
        params = {**default_params, **self.__params}

        # `start` needed for manual pagination
        # when API does not return paging info
        if start:
            params["start"] = start

        with self.__session() as session:
            try:
                response = session.get(self.__endpoint, params=params)
                response.raise_for_status()
                raw_data = response.json()
            except requests.HTTPError:
                raise

        # [0] object of json contain some metainfo
        try:
            result = raw_data[1]
        except BaseException:
            raise

        return result

    def __get_data(self):
        """Iterate throu pages to get them all.

        (MOEX API returns paginated data)
        """
        # set paging params
        start, total = 0, 0
        full_data = []

        page = self.__get_one_page(start)

        # if API support pagiantion
        if (self.__parsing_objest + ".cursor") in page:

            # set paging params
            start, total = 0, 0

            try:
                # get paging info from response
                paging = page[self.__parsing_objest + ".cursor"]

                # then remove paging object from response
                del page[self.__parsing_objest + ".cursor"]

                # parse data and add to result
                full_data.append(pd.DataFrame(page[self.__parsing_objest]))

                # get total rows
                total = paging[0]["TOTAL"]

                # calc next page
                start += paging[0]["PAGESIZE"]

            except BaseException:
                raise

            # iterate through other pages (if there's more than one)
            while start <= total:

                page = self.__get_one_page(start)

                try:
                    # get paging info from response
                    paging = page[self.__parsing_objest + ".cursor"]

                    # then remove paging object from response
                    del page[self.__parsing_objest + ".cursor"]

                    # parse data and add to result
                    full_data.append(pd.DataFrame(page[self.__parsing_objest]))

                    # calc next page
                    start += paging[0]["PAGESIZE"]
                except BaseException:
                    raise

        # else manual paging
        else:
            try:
                # parse data
                chunk = pd.DataFrame(
                    page[self.__parsing_objest])
            except BaseException:
                raise

            # add it to result
            full_data.append(chunk)

            # get size of page (number of records)
            chunk_size = len(chunk.index)

            # set start page for next call
            start = chunk_size

            # iterate through other pages, if there are any data in result
            while chunk_size:
                page = self.__get_one_page(start)

                try:
                    # parse data
                    chunk = pd.DataFrame(page[self.__parsing_objest])
                except BaseException:
                    raise

                # add it to result
                full_data.append(chunk)

                # get size of page
                chunk_size = len(chunk.index)

                # set start page for next call
                start += chunk_size

        return pd.concat(full_data)

    def get_ticker_history(self, ticker: str,
                           from_: str = None,
                           till: str = None,
                           columns: str = None):
        """Call 'ticker history' API.

        Parameters
        ----------
        ticker : must be in upcase, e.g. "SBER"

        from_ : must be in "YYYY-MM-DD" format

        till : "YYYY-MM-DD"

        columns: upcase without spaces, e.g. "TRADEDATE,OPEN,CLOSE"

        Return
        ------
        pd.Dataframe with ticker data

        See Also
        --------
        spec: https://iss.moex.com/iss/reference/63
        """
        self.__prepare_params(from_=from_, till=till, columns=columns)
        self.__parsing_objest = "history"
        market = "shares"
        engine = "stock"
        self.__endpoint = (f"https://iss.moex.com/iss/history/engines/{engine}"
                           f"/markets/{market}/"
                           f"securities/{ticker}.json")
        result = self.__get_data()

        result = result[result["BOARDID"].isin(["TQBR"])]

        return result

    def get_ticker_candles(self, ticker: str,
                           from_: str = None,
                           till: str = None,
                           columns: str = None,
                           interval: int = 60):
        """Call 'ticker candles' MOEX API.

        Parameters
        ----------
        ticker : must be in upcase, e.g. "SBER"

        from_: must be in "YYYY-MM-DD" format

        till : "YYYY-MM-DD"

        columns : upcase without spaces, e.g. "TRADEDATE,OPEN,CLOSE"

        interval :
            1(1 min),10(10 min),60(hour),24(day),7(week),31(month),4(quart)

        Return
        ------
        pd.Dataframe with ticker data

        See Also
        --------
        spec: https://iss.moex.com/iss/reference/46

        """
        self.__prepare_params(from_=from_, till=till,
                              columns=columns, interval=interval)
        self.__parsing_objest = "candles"
        market = "shares"
        engine = "stock"
        board = "TQBR"
        self.__endpoint = (f"https://iss.moex.com/iss/engines/{engine}"
                           f"/markets/{market}/"
                           f"boards/{board}/securities/{ticker}/candles.json")
        return self.__get_data()
