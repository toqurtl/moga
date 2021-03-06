class BinaryChromosomeBoundaryError(Exception):
    def __init__(self, bound_info):
        check = True
        min_value, max_value, num_digit = bound_info
        if min_value < 0 or max_value >= pow(2, num_digit):
            check = False
            print('min value have to be positive')
        elif max_value >= pow(2, num_digit):
            check = False
            print('max value have to be smaller than num_digit')
        else:
            check = True


class HyperParameterSettingError(Exception):
    def __init__(self):
        msg = "sum of each operator ratio have to be 1"
        super().__init__(msg)


class GenoTypeRangeException(Exception):
    def __init__(self, order, threshold, input_value):
        if order == 1:
            msg = 'min value have to be positive - '
            msg += 'threshold: ' + str(threshold) + '  input value: ' + str(input_value)
        else:
            msg = 'max value have to be smaller than num_digit - '
            msg += 'threshold: ' + str(threshold) + '  input value: ' + str(input_value)
        super().__init__(msg)