\begin{table}[t]
    \centering
    \scriptsize
    \renewcommand{\arraystretch}{1.15}
    \setlength{\tabcolsep}{4pt}

    \begin{tabularx}{\linewidth}{@{}%
        >{\raggedright\arraybackslash}p{0.26\linewidth}
        >{\centering\arraybackslash}X
        >{\centering\arraybackslash}p{0.16\linewidth}
    @{}}
    \toprule
    \textbf{Go2 AMP Reward Component} & \textbf{Definition} & \textbf{Scale} \\
    \midrule

    AMP style reward &
    $\displaystyle
    r_{\mathrm{style}} =
    \alpha_{\mathrm{AMP}}
    \max\!\left(
    1-\frac{1}{4}\bigl(D(s_t,s_{t+1})-1\bigr)^2,\,
    0
    \right)
    $ &
    $\displaystyle \alpha_{\mathrm{AMP}} = 2.0$ \\

    Task reward aggregation &
    $\displaystyle
    r_{\mathrm{task}} =
    \max\!\left(
    1.5\,r_{\mathrm{lin}}
    +0.5\,r_{\mathrm{yaw}}
    -3\times 10^{-4}\,c_{\mathrm{act}},
    \,0
    \right)
    $ &
    clipped at zero \\

    Final PPO reward &
    $\displaystyle
    r_{\mathrm{PPO}} =
    (1-\lambda_{\mathrm{task}})\,r_{\mathrm{style}}
    +\lambda_{\mathrm{task}}\,r_{\mathrm{task}}
    $ &
    $\displaystyle \lambda_{\mathrm{task}} = 0.3$ \\

    \bottomrule
    \end{tabularx}
    \caption{Reward used for the Unitree Go2 AMP velocity-tracking benchmark. Here $D(s_t,s_{t+1})$ is the AMP discriminator output and $r_{\mathrm{task}}$ is the environment task reward before AMP interpolation.}
    \label{tab:go2_amp_reward_composition}
\end{table}

\begin{table}[t]
    \centering
    \scriptsize
    \renewcommand{\arraystretch}{1.15}
    \setlength{\tabcolsep}{4pt}

    \begin{tabularx}{\linewidth}{@{}%
        >{\raggedright\arraybackslash}p{0.27\linewidth}
        >{\centering\arraybackslash}X
        >{\centering\arraybackslash}p{0.15\linewidth}
    @{}}
    \toprule
    \textbf{Go2 AMP Positive Task-Reward Term} & \textbf{Definition} & \textbf{Scale} \\
    \midrule

    Linear velocity tracking &
    $\displaystyle
    r_{\mathrm{lin}} =
    \exp\!\left(
    -\frac{
    \|\mathbf{v}^{\mathrm{cmd}}_{xy}
    -\mathbf{v}^{\mathrm{base}}_{xy}\|_2^2
    }{0.25}
    \right)
    $ &
    $\displaystyle w_{\mathrm{lin}} = 1.5$ \\

    Yaw-rate tracking &
    $\displaystyle
    r_{\mathrm{yaw}} =
    \exp\!\left(
    -\frac{
    (\omega_z^{\mathrm{cmd}}-\omega_z^{\mathrm{base}})^2
    }{0.25}
    \right)
    $ &
    $\displaystyle w_{\mathrm{yaw}} = 0.5$ \\

    \bottomrule
    \end{tabularx}
    \caption{Positive task-reward terms for Unitree Go2 AMP velocity tracking. The base-frame command ranges are $v_x^{\mathrm{cmd}}\in[0.0,0.5]$ m/s, $v_y^{\mathrm{cmd}}\in[-0.3,0.3]$ m/s, and $\omega_z^{\mathrm{cmd}}\in[-1.57,1.57]$ rad/s. Commands are resampled every $10$ s and smoothed as $\mathbf{u}_t=0.99\mathbf{u}_{t-1}+0.01\tilde{\mathbf{u}}_t$. The listed weights are runtime weights after the environment multiplies nonzero reward scales by the policy timestep $\Delta t=0.005\times6=0.03$ s.}
    \label{tab:go2_amp_reward_positive}
\end{table}

\begin{table}[t]
    \centering
    \scriptsize
    \renewcommand{\arraystretch}{1.15}
    \setlength{\tabcolsep}{4pt}

    \begin{tabularx}{\linewidth}{@{}%
        >{\raggedright\arraybackslash}p{0.27\linewidth}
        >{\centering\arraybackslash}X
        >{\centering\arraybackslash}p{0.15\linewidth}
    @{}}
    \toprule
    \textbf{Go2 AMP Task Penalty Term} & \textbf{Definition} & \textbf{Scale} \\
    \midrule

    Action rate penalty &
    $\displaystyle
    c_{\mathrm{act}} =
    \|\mathbf{a}_t-\mathbf{a}_{t-1}\|_2^2
    $ &
    $\displaystyle w_{\mathrm{act}} = -3\times10^{-4}$ \\

    \bottomrule
    \end{tabularx}
    \caption{Nonzero task penalty for the Unitree Go2 AMP velocity-tracking benchmark. Termination, torque, joint-acceleration, collision, feet-air-time, and joint-limit terms are configured with zero scale in this AMP benchmark and are omitted.}
    \label{tab:go2_amp_reward_penalty}
\end{table}

\begin{table}[t]
    \centering
    \small
    \setlength{\tabcolsep}{4pt}
    \caption{Domain randomization and stochastic training parameters for the Unitree Go2 AMP benchmark.}
    \label{tab:go2_amp_domain_randomization}
    \begin{tabular}{@{}lll@{}}
    \toprule
    Category & Parameter & Range / Value \\
    \midrule
    Contact & Tangential friction & $[0.25,\,1.75]$ \\
    Base dynamics & Added base mass & $[-1.0,\,1.0]$ kg \\
    Control & $K_p$ scale & $[0.9,\,1.1]$ \\
    Control & $K_d$ scale & $[0.9,\,1.1]$ \\
    Observation noise & Base linear velocity, critic only & $\pm 0.1$ m/s \\
    Observation noise & Base angular velocity & $\pm 0.3$ rad/s \\
    Observation noise & Gravity estimate & $\pm 0.05$ \\
    Observation noise & Joint position & $\pm 0.03$ rad \\
    Observation noise & Joint velocity & $\pm 1.5$ rad/s \\
    Pushes & Interval & $15.0$ s \\
    Pushes & Linear velocity reset $(x,y)$ & $[-1.0,\,1.0]$, $[-1.0,\,1.0]$ m/s \\
    Initialization & Reference-state initialization & Mocap state with probability $0.85$ \\
    \bottomrule
    \end{tabular}
\end{table}
