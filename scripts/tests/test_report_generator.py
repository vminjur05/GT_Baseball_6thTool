from scripts.report_generator import ReportGenerator
import pandas as pd
import numpy as np

def run_test():
    n = 200
    df = pd.DataFrame({
        'PitchVelo': np.random.normal(87,4,size=n),
        'ExitVelo': np.random.normal(85,6,size=n),
        'LaunchAng': np.random.normal(18,8,size=n),
        'BallInPlay': np.random.choice([True, False], size=n, p=[0.4,0.6]),
        'Inning': np.random.choice([1,2,3,4,5,6,7,8,9], size=n),
        'PitchOutcome': np.random.choice(['Strike','Ball','Foul','InPlay'], size=n),
        'PitcherName': np.random.choice(['P1','P2','P3','P4','P5'], size=n),
        'FielderRouteEfficiency': np.random.normal(85,8,size=n),
        'FielderReaction': np.random.normal(1.2,0.3,size=n),
        'BatterTimeToFirst': np.random.normal(4.3,0.4,size=n),
        'BaserunnerMaxSpeed': np.random.normal(18,1.5,size=n)
    })

    rg = ReportGenerator(df, report_dir='reports')
    pdf_bytes = rg.export_pdf_bytes()
    print('export_pdf_bytes returned length:', len(pdf_bytes))

    rg.export_pdf('test_weekly_report.pdf')
    print('export_pdf wrote file to reports/test_weekly_report.pdf')

if __name__ == '__main__':
    run_test()
