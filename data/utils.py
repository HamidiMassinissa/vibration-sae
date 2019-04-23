import datetime


def float_of_strDateTime(strDateTime):
    a = datetime.datetime.strptime(strDateTime, "%d/%m/%Y %H:%M:%S")
    return a.timestamp()
