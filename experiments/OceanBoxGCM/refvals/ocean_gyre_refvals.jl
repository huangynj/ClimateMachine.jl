# Testing reference values and precisions
# Each test block of varr and parr should be followed by an append to refVals, refPrecs arrays.
# e.g.
#   refVals=[]
#   refPrecs=[]
#
#   varr = ..........
#   par  = ..........
#
#   append!(refVals ,[ varr ] )
#   append!(refPrecs,[ parr ] )
#
#   varr = ..........
#   par  = ..........
#
#   append!(refVals ,[ varr ] )
#   append!(refPrecs,[ parr ] )
#
#   varr = ..........
#   par  = ..........
#
#   append!(refVals ,[ varr ] )
#   append!(refPrecs,[ parr ] )
#
#   etc.....
#
#   Now for real!
#

refVals = []
refPrecs = []

# SC ========== Test number 1 reference values and precision match template. =======
# SC ========== /home/cnh/projects/cm/experiments/OceanBoxGCM/ocean_gyre.jl test reference values ======================================
# BEGIN SCPRINT
# varr - reference values (from reference run)
# parr - digits match precision (hand edit as needed)
#
# [
#  [ MPIStateArray Name, Field Name, Maximum, Minimum, Mean, Standard Deviation ],
#  [         :                :          :        :      :          :           ],
# ]
varr = [
    [
        "Q",
        "u[1]",
        -7.88803234115370288659e-02,
        6.54328011157667283060e-02,
        2.01451525264660777012e-03,
        1.24588716783015467093e-02,
    ],
    [
        "Q",
        "u[2]",
        -8.71605274310934263760e-02,
        1.47505267452495314462e-01,
        5.63158201082454307890e-03,
        1.28959719954838437916e-02,
    ],
    [
        "Q",
        :η,
        -4.73491408441974459542e-01,
        4.02693687285993751068e-01,
        -6.49059970676089264836e-05,
        2.21689928090662985438e-01,
    ],
    [
        "Q",
        :θ,
        4.24292935192304451839e-04,
        9.24539353455580759089e+00,
        2.49938206627401893201e+00,
        2.17986626392490689952e+00,
    ],
    [
        "s_aux",
        :y,
        0.00000000000000000000e+00,
        4.00000000000000046566e+06,
        1.99999999999999976717e+06,
        1.15573163901915703900e+06,
    ],
    [
        "s_aux",
        :w,
        -2.22086406767055965930e-04,
        2.00575090959195941574e-04,
        2.53168866096191178104e-07,
        1.66257132341925427171e-05,
    ],
    [
        "s_aux",
        :pkin,
        -9.00869877619915215838e-01,
        0.00000000000000000000e+00,
        -3.33171488779369640021e-01,
        2.54740287894525019308e-01,
    ],
    [
        "s_aux",
        :wz0,
        -2.96608015794916532427e-05,
        3.66759042312538147113e-05,
        3.78116102589123325624e-10,
        1.07073256826409470244e-05,
    ],
    [
        "s_aux",
        "uᵈ[1]",
        0.00000000000000000000e+00,
        0.00000000000000000000e+00,
        0.00000000000000000000e+00,
        0.00000000000000000000e+00,
    ],
    [
        "s_aux",
        "uᵈ[2]",
        0.00000000000000000000e+00,
        0.00000000000000000000e+00,
        0.00000000000000000000e+00,
        0.00000000000000000000e+00,
    ],
    [
        "s_aux",
        "ΔGᵘ[1]",
        0.00000000000000000000e+00,
        0.00000000000000000000e+00,
        0.00000000000000000000e+00,
        0.00000000000000000000e+00,
    ],
    [
        "s_aux",
        "ΔGᵘ[2]",
        0.00000000000000000000e+00,
        0.00000000000000000000e+00,
        0.00000000000000000000e+00,
        0.00000000000000000000e+00,
    ],
]
parr = [
    ["Q", "u[1]", 12, 12, 12, 12],
    ["Q", "u[2]", 12, 12, 12, 12],
    ["Q", :η, 12, 12, 12, 12],
    ["Q", :θ, 12, 12, 12, 12],
    ["s_aux", :y, 12, 12, 12, 12],
    ["s_aux", :w, 12, 12, 12, 12],
    ["s_aux", :pkin, 12, 12, 12, 12],
    ["s_aux", :wz0, 12, 12, 8, 12],
    ["s_aux", "uᵈ[1]", 12, 12, 12, 12],
    ["s_aux", "uᵈ[2]", 12, 12, 12, 12],
    ["s_aux", "ΔGᵘ[1]", 12, 12, 12, 12],
    ["s_aux", "ΔGᵘ[2]", 12, 12, 12, 12],
]
# END SCPRINT
# SC ====================================================================================

append!(refVals, [varr])
append!(refPrecs, [parr])

# SC ========== Test number 2 reference values and precision match template. =======
# SC ========== /home/cnh/projects/cm/experiments/OceanBoxGCM/ocean_gyre.jl test reference values ======================================
# BEGIN SCPRINT
# varr - reference values (from reference run)
# parr - digits match precision (hand edit as needed)
#
# [
#  [ MPIStateArray Name, Field Name, Maximum, Minimum, Mean, Standard Deviation ],
#  [         :                :          :        :      :          :           ],
# ]
varr = [
    [
        "Q",
        "u[1]",
        -3.08415268094245687003e-01,
        3.31683879748177812274e-01,
        2.34355645025035056947e-03,
        1.38278493625595393784e-02,
    ],
    [
        "Q",
        "u[2]",
        -9.30330187375882494694e-02,
        1.45773331490709479041e-01,
        6.38739141648009701308e-03,
        1.38107523198905141754e-02,
    ],
    [
        "Q",
        :η,
        -4.60215148246349181615e-01,
        3.97155090993862369686e-01,
        -1.05814653709044253051e-04,
        2.14356824554893188317e-01,
    ],
    [
        "Q",
        :θ,
        3.90305009874997816885e-04,
        9.28488770009523101123e+00,
        2.49938688339290848717e+00,
        2.17986650147491722862e+00,
    ],
    [
        "s_aux",
        :y,
        0.00000000000000000000e+00,
        4.00000000000000046566e+06,
        1.99999999999999976717e+06,
        1.15573163901915703900e+06,
    ],
    [
        "s_aux",
        :w,
        -2.65432487944419423249e-04,
        2.51335486222488370053e-04,
        3.34985017281366756292e-07,
        1.98256782258218657716e-05,
    ],
    [
        "s_aux",
        :pkin,
        -9.01203660410079376852e-01,
        0.00000000000000000000e+00,
        -3.33173652152760846334e-01,
        2.54737286187841360796e-01,
    ],
    [
        "s_aux",
        :wz0,
        -3.16776291712146345464e-05,
        3.89428424525183031852e-05,
        4.05688221636862481177e-10,
        1.10561449663397378925e-05,
    ],
    [
        "s_aux",
        "uᵈ[1]",
        0.00000000000000000000e+00,
        0.00000000000000000000e+00,
        0.00000000000000000000e+00,
        0.00000000000000000000e+00,
    ],
    [
        "s_aux",
        "uᵈ[2]",
        0.00000000000000000000e+00,
        0.00000000000000000000e+00,
        0.00000000000000000000e+00,
        0.00000000000000000000e+00,
    ],
    [
        "s_aux",
        "ΔGᵘ[1]",
        0.00000000000000000000e+00,
        0.00000000000000000000e+00,
        0.00000000000000000000e+00,
        0.00000000000000000000e+00,
    ],
    [
        "s_aux",
        "ΔGᵘ[2]",
        0.00000000000000000000e+00,
        0.00000000000000000000e+00,
        0.00000000000000000000e+00,
        0.00000000000000000000e+00,
    ],
]
parr = [
    ["Q", "u[1]", 12, 12, 12, 12],
    ["Q", "u[2]", 12, 12, 12, 12],
    ["Q", :η, 12, 12, 12, 12],
    ["Q", :θ, 12, 12, 12, 12],
    ["s_aux", :y, 12, 12, 12, 12],
    ["s_aux", :w, 12, 12, 12, 12],
    ["s_aux", :pkin, 12, 12, 12, 12],
    ["s_aux", :wz0, 12, 12, 8, 12],
    ["s_aux", "uᵈ[1]", 12, 12, 12, 12],
    ["s_aux", "uᵈ[2]", 12, 12, 12, 12],
    ["s_aux", "ΔGᵘ[1]", 12, 12, 12, 12],
    ["s_aux", "ΔGᵘ[2]", 12, 12, 12, 12],
]
# END SCPRINT
# SC ====================================================================================

append!(refVals, [varr])
append!(refPrecs, [parr])
