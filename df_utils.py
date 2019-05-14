from logger import logger


class SeriesStats:
    def __init__(self, header, dtype, nunique, nans):
        self.header = header
        self.dtype = dtype
        self.nunique = nunique
        self.nans = nans

    @staticmethod
    def display_title():
        logger.info("Header\t: Dtype\t Nunique\t NaNs")

    def display(self):
        logger.info("{}\t: {}\t {}\t {}".format(self.header, self.dtype, self.nunique, self.nans))

    @staticmethod
    def show_delta(header, series_a, series_b):
        dtype_a = series_a.dtype
        dtype_b = series_b.dtype
        if dtype_a.base != dtype_b.base:
            logger.info("... {} dtype changed: {} -> {}".format(header, dtype_a, dtype_b))

        nunique_a = series_a.nunique
        nunique_b = series_b.nunique
        if nunique_a != nunique_b:
            logger.info("... {} nunique changed: {} -> {}".format(header, nunique_a, nunique_b))

        nan_a = series_a.nans
        nan_b = series_b.nans
        if nan_a != nan_b:
            logger.info("... {} nan_b changed: {} -> {}".format(header, nan_a, nan_b))

class DataframeStats:
    def __init__(self, shape, header_list, series_dict):
        self.shape = shape
        self.header_list = header_list
        self.series_dict = series_dict

    @staticmethod
    def snapshot(df):
        shape = df.shape
        series_dict = dict()

        header_list = df.columns.values
        for header in header_list:
            nunique = df[header].nunique()
            dtype = df[header].dtype
            nans = df[header].isna().sum()
            series_dict[header] = SeriesStats(header, dtype, nunique, nans)

        df_snap = DataframeStats(shape, header_list, series_dict)
        return df_snap

    def display(self):
        logger.info("Shape: {}".format(self.shape))
        SeriesStats.display_title()
        for header in self.header_list:
            series_stats = self.series_dict[header]
            series_stats.display()

    @staticmethod
    def show_delta(snapshot_a, shapshot_b):
        rows_before = snapshot_a.shape[0]
        cols_before = snapshot_a.shape[1]
        rows_after = shapshot_b.shape[0]
        cols_after = shapshot_b.shape[1]
        row_diff = rows_before - rows_after
        col_diff = cols_before - cols_after

        if row_diff > 0:
            logger.info("... records removed: {} ({:.4%} of total)".format(row_diff, (1*row_diff/rows_before)))
        elif row_diff < 0:
            row_diff = - row_diff
            logger.info("... records added: {} ({:.4%} of total)".format(row_diff,  (1*row_diff/rows_before)))

        if col_diff > 0:
            logger.info("... columns removed: {} ({:.4%} of total)".format(col_diff, (1*col_diff/cols_before)))
        elif col_diff < 0:
            col_diff = - col_diff
            logger.info("... columns added: {} ({:.4%} of total)".format(col_diff, (1*col_diff/cols_before)))

        if row_diff == 0:
            for header in snapshot_a.header_list:
                if header not in shapshot_b.header_list:
                    return
                else:
                    series_a = snapshot_a.series_dict[header]
                    series_b = shapshot_b.series_dict[header]
                    SeriesStats.show_delta(header, series_a, series_b)
