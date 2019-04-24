from config import Configuration


if __name__ == '__main__':

    config = Configuration()
    config.parse_commandline()
    config.new_experiment()

    if config.RUN == 'bayesopt':
        from opt.bayesopt import main
        main()
    if config.RUN == 'monitoring':
        pass
    if config.RUN == 'crossval':
        from validation.test import runCV
        runCV()
    if config.RUN == 'walkforward':
        pass
