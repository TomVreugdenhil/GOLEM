#%matplotlib inline
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants, integrate, signal, interpolate
import sqlalchemy   # high-level library for SQL in Python
import pandas as pd
import subprocess
#Interactive web figures
import holoviews as hv
hv.extension('bokeh')
import hvplot.pandas
#conditional rich-text boxes
from IPython.display import Markdown

# Define Global constants
data_URL = "http://golem.fjfi.cvut.cz/shots/{shot_no}/Diagnostics/BasicDiagnostics/{identifier}"  # TODO workaround
parameters_URL = 'http://golem.fjfi.cvut.cz/shots/{shot_no}/Production/Parameters/{identifier}'

# SUPPORTING FUNCTIONS
def print_and_save(phys_quant, value ,dest, format_str='%.3f'):
    print(phys_quant+" = %.5f" % value)
    with open(dest+phys_quant, 'w') as f:
        f.write(format_str % value)
    update_db_current_shot(phys_quant,value)

def update_db_current_shot_alchemy(field_name, value):
    try:
        engine = sqlalchemy.create_engine('postgresql://golem@/golem_database?host=/var/run/postgresql')
    except:
        return
    engine.execute(f"""UPDATE shots SET "{field_name}"={value} WHERE shot_no IN(SELECT max(shot_no) FROM shots)""")

def update_db_current_shot(field_name, value):
    #os.system('psql -c "UPDATE shots SET '+field_name+'='+str(value)+' WHERE shot_no IN(SELECT max(shot_no) FROM shots)" -q -U golem golem_database')
    subprocess.call(["export PGPASSWORD='rabijosille';psql -q -U golem golem_database --command='UPDATE shots SET \""+field_name+"\"="+str(value)+" WHERE shot_no IN(SELECT max(shot_no) FROM shots)'"],shell=True)


def open_remote(shot_no, identifier, url_template=data_URL):
    #Datasource create tmp files
    #ds = np.DataSource()
    return np.DataSource().open(url_template.format(shot_no=shot_no, identifier=identifier))

def read_value(shot_no, identifier):
    """Return the value for given shot as a number if possible"""
    value = open_remote(shot_no, identifier, data_URL).read()
    return pd.to_numeric(value, errors='ignore')

def read_parameter(shot_no, identifier):
    return open_remote(shot_no, identifier, parameters_URL).read().strip()

def read_signal(shot_no, identifier):
    file = open_remote(shot_no, identifier, data_URL + '.csv')
    return pd.read_csv(file, names=['Time',identifier],
                     index_col='Time', squeeze=True)  # squeeze makes simple 1-column signals a Series

def correct_inf(signal):
    """Inteprolate Inf values"""
    signal = signal.replace([-np.inf, np.inf], np.nan).interpolate()
    return signal

def show_plasma_limits(is_plasma,t_plasma_start, t_plasma_end,in_seconds=True):
    t_scale = 1e-3 if in_seconds else 1
    if is_plasma:
        for t in (t_plasma_start, t_plasma_end):
            plt.axvline(t * t_scale, color='k', linestyle='--')

# ACTUAL FUNCTION
def DiagProcess(shot_no):



    destination=f'Results/{shot_no}'
    if not os.path.exists(destination):
        os.mkdir(destination)
    source = 'Results/'
#os.makedirs(destination, exist_ok=True );
# try to get thot number form SHOT_NO envirnoment variable, otherwise use the specified one
    #shot_no = os.environ.get('SHOT_NO', 0) # not using this



    t_Bt = float(read_parameter(shot_no, 'TBt')) * 1e-6  # from us to s
    t_CD = float(read_parameter(shot_no, 'Tcd')) * 1e-6  # from us to s
    offset_sl = slice(None, min(t_Bt, t_CD) - 1e-4)
# Plasma detection result loading
    is_plasma = read_value(shot_no, source+'is_plasma') == 1
    t_plasma_start = read_value(shot_no, source+'t_plasma_start')
    t_plasma_end = read_value(shot_no, source+'t_plasma_end')-10**(-3)
    plasma_lifetime = read_value(shot_no, source+'t_plasma_duration')
    if is_plasma:
        heading = Markdown("### Plasma detected\n\n"
    f"plasma lifetime of {plasma_lifetime:.1f} ms, from {t_plasma_start:.1f} ms to {t_plasma_end:.1f} ms")
    else:
        heading = Markdown("### No plasma detected (vacuum discharge)")
    heading

# U_I management
    loop_voltage = read_signal(shot_no, 'U_Loop')
    polarity_CD = read_parameter(shot_no, 'CD_orientation')
    if polarity_CD != 'CW':                   # TODO hardcoded for now!
        loop_voltage *= -1  # make positive
    loop_voltage = correct_inf(loop_voltage)
    loop_voltage.loc[:t_CD] = 0
    ax = loop_voltage.plot(grid=True)
    show_plasma_limits(is_plasma,t_plasma_start, t_plasma_end)
    ax.set(xlabel="Time [s]", ylabel="$U_l$ [V]", title="Loop voltage $U_l$ #{}".format(shot_no));
# B_t calculation
    dBt = read_signal(shot_no,'U_BtCoil')
    polarity_Bt = read_parameter(shot_no, 'Bt_orientation')
    if polarity_Bt != 'CW':                   # TODO hardcoded for now!
        dBt *= -1  # make positive
    dBt = correct_inf(dBt)
    dBt -= dBt.loc[offset_sl].mean()
    ax = dBt.plot(grid=True)
    show_plasma_limits(is_plasma,t_plasma_start, t_plasma_end)
    ax.set(xlabel="Time [s]", ylabel="$dU_{B_t}/dt$ [V]", title="BtCoil_raw signal #{}".format(shot_no));
#  Integration (it is a magnetic diagnostic) & calibration
    K_BtCoil = float(read_parameter(shot_no, 'SystemParameters/K_BtCoil')) # Get BtCoil calibration factor
    print('BtCoil calibration factor K_BtCoil={} T/(Vs)'.format(K_BtCoil))

    Bt = pd.Series(integrate.cumtrapz(dBt, x=dBt.index, initial=0) * K_BtCoil,
                   index=dBt.index, name='Bt')
    ax = Bt.plot(grid=True)
    show_plasma_limits(is_plasma,t_plasma_start, t_plasma_end)
    ax.set(xlabel="Time [s]", ylabel="$B_t$ [T]", title="Toroidal magnetic field $B_t$ #{}".format(shot_no));

# Chamber +plasma current calculation
    dIpch = read_signal(shot_no, 'U_RogCoil')
    if polarity_CD != 'CW':                   # TODO hardcoded for now!
        dIpch *= -1  # make positive
    dIpch = correct_inf(dIpch)
    dIpch -= dIpch.loc[offset_sl].mean() # subtract offset
    dIpch.loc[:t_CD] = 0
    ax = dIpch.plot(grid=True)
    show_plasma_limits(is_plasma,t_plasma_start, t_plasma_end)
    ax.set(xlabel="Time [s]", ylabel="$dU_{I_{p+ch}}/dt$ [V]", title="RogowskiCoil_raw signal #{}".format(shot_no));

# Integration & calibration
    K_RogowskiCoil = float(read_parameter(shot_no, 'SystemParameters/K_RogowskiCoil')) # Get RogowskiCoil calibration factor
    print('RogowskiCoil calibration factor K_RogowskiCoil={} A/(Vs)'.format(K_RogowskiCoil))

    Ipch = pd.Series(integrate.cumtrapz(dIpch, x=dIpch.index, initial=0) * K_RogowskiCoil,
                    index=dIpch.index, name='Ipch')
    ax = Ipch.plot(grid=True)
    show_plasma_limits(is_plasma,t_plasma_start, t_plasma_end)
    ax.set(xlabel="Time [s]", ylabel="$I_{p+ch}$ [A]", title="Total (plasma+chamber) current $I_{{p+ch}}$ #{}".format(shot_no));

#Chamber current I_ch calculation
    R_chamber = float(read_parameter(shot_no, 'SystemParameters/R_chamber')) # Get Chamber resistivity
    print('Chamber resistivity R_chamber={} Ohm'.format(R_chamber))
    L_chamber = float(read_parameter(shot_no, 'SystemParameters/L_chamber')) # Get Chamber inductance
    print('Chamber inductance L_chamber={} H'.format(L_chamber))


    U_l_func = interpolate.interp1d(loop_voltage.index, loop_voltage)  # 1D interpolator
    def dIch_dt(t, Ich):
        return (U_l_func(t) - R_chamber * Ich) / L_chamber
    t_span = loop_voltage.index[[0, -1]]
    solution = integrate.solve_ivp(dIch_dt, t_span, [0], t_eval=loop_voltage.index, )
    Ich = pd.Series(solution.y[0], index=loop_voltage.index, name='Ich')

    for I in [Ich.rename('$I_{ch}$'), Ipch.rename('$I_{ch}(+I_p)$')]:
        ax = I.plot()
    ax.legend()
    show_plasma_limits(is_plasma,t_plasma_start, t_plasma_end)
    ax.set(xlabel='Time [s]', ylabel='$I$ [A]', title='estimated chamber current and measured total')
    plt.grid()

#Plasma current I_p calculation

    if is_plasma:
        Ip_naive = Ipch - loop_voltage/R_chamber  # creates a new Series
        Ip = Ipch - Ich
        Ip.name = 'Ip'
        Ip_naive.plot(grid=True, label='naive $I_{ch}=U_l/R_{ch}$')
        ax = Ip.plot(grid=True, label=r'using $U_l = R_{ch} I_{ch} - L_{ch} \frac{d I_{ch}}{dt}$')
        ax.legend()
        show_plasma_limits(is_plasma,t_plasma_start, t_plasma_end)
        ax.set(xlabel="Time [s]", ylabel="$I_{p}$ [A]", title="Plasma current $I_{{p}}$ #{}".format(shot_no));
    else:
        Ip = Ipch * 0  # no current
    heading



    fig = plt.figure(dpi=200)
    for I in [Ich.rename('$I_{ch}$'), Ipch.rename('$I_{ch}+I_p$'), Ip.rename('$I_p$')]:
        ax = I.plot()
    ax.legend()
    show_plasma_limits(is_plasma,t_plasma_start, t_plasma_end)
    ax.set(xlabel='Time [s]', ylabel='$I$ [A]', title='estimated plasma and chamber current and measured total')
    plt.grid()
    plt.savefig('icon-fig.png')
# Overview graphs and parameters
    df_processed = pd.concat(
        [loop_voltage.rename('U_loop'), Bt, Ip*1e-3, Ich*1e-3], axis='columns')
    df_processed.index = df_processed.index * 1e3  # to ms
    df_processed.head()

    if is_plasma:
        plasma_lines = hv.VLine(t_plasma_start) * hv.VLine(t_plasma_end)
        Ip_line = df_processed['Ip'].hvplot.line(ylabel='Iᴄʜ, Iₚ [kA]', label='Iₚ', by=[], xlabel='time [ms]')
    else:
        plasma_lines = Ip_line = hv.Curve([])
    layout = df_processed['U_loop'].hvplot.line(ylabel='Uₗ [V]', xlabel='', by=[]) * plasma_lines +\
    df_processed['Bt'].hvplot.line(ylabel='Bₜ [T]', xlabel='', by=[]) * plasma_lines +\
    df_processed['Ich'].hvplot.line(label='Iᴄʜ', by=[]) * Ip_line *\
       plasma_lines

    plot = layout.cols(1).opts(
        hv.opts.Curve(width=600, height=200, title='', ylim=(0, None), show_grid=True),
        hv.opts.VLine(color='black', alpha=0.7, line_dash='dashed')
                            )
    hvplot.save(plot, 'homepage_figure.html')
    plot

    signal_files = []
    for sig_name, signal in df_processed.items():
        fname = f'{destination}/{sig_name}.csv'
        signal.to_csv(fname, header=False)
        signal_files.append(fname)

    units = ['V', 'T', 'kA', 'kA']

    Markdown("Time series in graph in CSV format:\n"
             + "\n".join(f' - [{fn.split("/")[-1]}]({fn}) [ms, {u}]'
                         for (u, fn) in zip(units, signal_files)))

    if is_plasma:
        plasma_sl = slice(t_plasma_start, t_plasma_end)
    else:
        plasma_sl = slice(t_Bt, None)   # TODO really use whole discharge ?
    df_during_plasma = df_processed.loc[plasma_sl]
    df_overview = df_during_plasma.quantile([0.01, 0.5, 0.99])  # use quantiles to skip peaks
    df_overview.index = ['min', 'mean', 'max']  # actually quantiles, but easier to understand
    if is_plasma:
        df_overview.loc['start'] = df_during_plasma.iloc[0]
        df_overview.loc['end'] = df_during_plasma.iloc[-1]
    else:
        df_overview.loc['start'] = df_overview.loc['end'] = np.nan
    df_overview.loc['units'] = units
    # make units row first
    df_overview = df_overview.iloc[np.roll(np.arange(df_overview.shape[0]), 1)]
    df_overview
    for agg in ('mean', 'max'):
        for quantity, value in df_overview.loc[agg].iteritems():
            print_and_save(quantity+'_'+agg, value,destination)
    print_and_save('U_loop_breakdown', df_overview.loc['start', 'U_loop'],destination)
    print_and_save('t_Ip_max', df_during_plasma.Ip.idxmax(),destination)
