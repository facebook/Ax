/*
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 */

function getPlotData() {
  const arm_data = {
    metrics: ['metric_a'],
    in_sample: {
      '0_0': {
        name: '0_0',
        parameters: {x1: 2.5, x2: 7.5},
        y: {metric_a: 24.129964413622268},
        y_hat: {metric_a: 23.954282593022686},
        se: {metric_a: 2.0},
        se_hat: {metric_a: 1.629655093167409},
        context_stratum: null,
      },
      '0_1': {
        name: '0_1',
        parameters: {x1: 6.25, x2: 3.75},
        y: {metric_a: 26.624171220014905},
        y_hat: {metric_a: 26.487960178840346},
        se: {metric_a: 2.0},
        se_hat: {metric_a: 1.6726522245926425},
        context_stratum: null,
      },
      '0_10': {
        name: '0_10',
        parameters: {x1: 1.5625, x2: 8.4375},
        y: {metric_a: 31.321658517480987},
        y_hat: {metric_a: 31.206781430356646},
        se: {metric_a: 2.0},
        se_hat: {metric_a: 1.7424616887200985},
        context_stratum: null,
      },
      '0_11': {
        name: '0_11',
        parameters: {x1: -0.3125, x2: 2.8125},
        y: {metric_a: 32.8083830521407},
        y_hat: {metric_a: 33.08446828896634},
        se: {metric_a: 2.0},
        se_hat: {metric_a: 1.6649131283893672},
        context_stratum: null,
      },
      '0_12': {
        name: '0_12',
        parameters: {x1: 7.1875, x2: 10.3125},
        y: {metric_a: 98.34760790686913},
        y_hat: {metric_a: 98.48545144809549},
        se: {metric_a: 2.0},
        se_hat: {metric_a: 1.7465819092856345},
        context_stratum: null,
      },
      '0_13': {
        name: '0_13',
        parameters: {x1: 3.4375, x2: 6.5625},
        y: {metric_a: 21.127854002025494},
        y_hat: {metric_a: 21.328860131845442},
        se: {metric_a: 2.0},
        se_hat: {metric_a: 1.7597697952824811},
        context_stratum: null,
      },
      '0_14': {
        name: '0_14',
        parameters: {x1: -4.0625, x2: 14.0625},
        y: {metric_a: 4.47623958200588},
        y_hat: {metric_a: 4.700647450830054},
        se: {metric_a: 2.0},
        se_hat: {metric_a: 1.968364448184855},
        context_stratum: null,
      },
      '0_15': {
        name: '0_15',
        parameters: {x1: -3.59375, x2: 7.03125},
        y: {metric_a: 41.77178986158392},
        y_hat: {metric_a: 43.77522497690436},
        se: {metric_a: 2.0},
        se_hat: {metric_a: 1.7579263897550719},
        context_stratum: null,
      },
      '0_16': {
        name: '0_16',
        parameters: {x1: 3.90625, x2: 14.53125},
        y: {metric_a: 166.3236985012143},
        y_hat: {metric_a: 165.46807321090958},
        se: {metric_a: 2.0},
        se_hat: {metric_a: 1.9514975966309454},
        context_stratum: null,
      },
      '0_17': {
        name: '0_17',
        parameters: {x1: 7.65625, x2: 3.28125},
        y: {metric_a: 15.473497571024838},
        y_hat: {metric_a: 15.893875534568402},
        se: {metric_a: 2.0},
        se_hat: {metric_a: 1.8327004174255472},
        context_stratum: null,
      },
      '0_18': {
        name: '0_18',
        parameters: {x1: 0.15625, x2: 10.78125},
        y: {metric_a: 44.753611366285824},
        y_hat: {metric_a: 45.02461204157245},
        se: {metric_a: 2.0},
        se_hat: {metric_a: 1.8044890392202935},
        context_stratum: null,
      },
      '0_19': {
        name: '0_19',
        parameters: {x1: 2.03125, x2: 1.40625},
        y: {metric_a: 9.320218587634006},
        y_hat: {metric_a: 9.414259231158432},
        se: {metric_a: 2.0},
        se_hat: {metric_a: 1.5785591394402239},
        context_stratum: null,
      },
      '0_2': {
        name: '0_2',
        parameters: {x1: -1.25, x2: 11.25},
        y: {metric_a: 22.383482484999874},
        y_hat: {metric_a: 22.4229539561332},
        se: {metric_a: 2.0},
        se_hat: {metric_a: 1.6653050077281752},
        context_stratum: null,
      },
      '0_20': {
        name: '0_20',
        parameters: {x1: 9.53125, x2: 8.90625},
        y: {metric_a: 40.64753379393727},
        y_hat: {metric_a: 40.5704623207798},
        se: {metric_a: 2.0},
        se_hat: {metric_a: 1.8769515669300412},
        context_stratum: null,
      },
      '0_21': {
        name: '0_21',
        parameters: {x1: 5.78125, x2: 5.15625},
        y: {metric_a: 34.736739987996074},
        y_hat: {metric_a: 34.475989925184955},
        se: {metric_a: 2.0},
        se_hat: {metric_a: 1.5146845129936402},
        context_stratum: null,
      },
      '0_22': {
        name: '0_22',
        parameters: {x1: -1.71875, x2: 12.65625},
        y: {metric_a: 21.110094161886266},
        y_hat: {metric_a: 20.846900541617153},
        se: {metric_a: 2.0},
        se_hat: {metric_a: 1.5303026913453948},
        context_stratum: null,
      },
      '0_23': {
        name: '0_23',
        parameters: {x1: -2.65625, x2: 2.34375},
        y: {metric_a: 78.863835999175},
        y_hat: {metric_a: 79.80967320564336},
        se: {metric_a: 2.0},
        se_hat: {metric_a: 1.824779103586041},
        context_stratum: null,
      },
      '0_24': {
        name: '0_24',
        parameters: {x1: 4.84375, x2: 9.84375},
        y: {metric_a: 83.88052752898044},
        y_hat: {metric_a: 83.57547388294162},
        se: {metric_a: 2.0},
        se_hat: {metric_a: 1.830423983538325},
        context_stratum: null,
      },
      '0_25': {
        name: '0_25',
        parameters: {x1: 8.59375, x2: 6.09375},
        y: {metric_a: 21.424386764652148},
        y_hat: {metric_a: 21.229231411310828},
        se: {metric_a: 2.0},
        se_hat: {metric_a: 1.8954026948393508},
        context_stratum: null,
      },
      '0_26': {
        name: '0_26',
        parameters: {x1: 1.09375, x2: 13.59375},
        y: {metric_a: 98.68064045456067},
        y_hat: {metric_a: 98.14689670208656},
        se: {metric_a: 2.0},
        se_hat: {metric_a: 1.9505162703397199},
        context_stratum: null,
      },
      '0_27': {
        name: '0_27',
        parameters: {x1: -0.78125, x2: 4.21875},
        y: {metric_a: 26.449512503021804},
        y_hat: {metric_a: 26.36713210102177},
        se: {metric_a: 2.0},
        se_hat: {metric_a: 1.6278514132828605},
        context_stratum: null,
      },
      '0_28': {
        name: '0_28',
        parameters: {x1: 6.71875, x2: 11.71875},
        y: {metric_a: 130.64996148022834},
        y_hat: {metric_a: 130.55049030644375},
        se: {metric_a: 2.0},
        se_hat: {metric_a: 1.680291322781174},
        context_stratum: null,
      },
      '0_29': {
        name: '0_29',
        parameters: {x1: 2.96875, x2: 0.46875},
        y: {metric_a: 4.323605035294334},
        y_hat: {metric_a: 4.4251498301327175},
        se: {metric_a: 2.0},
        se_hat: {metric_a: 1.8448338972660463},
        context_stratum: null,
      },
      '0_3': {
        name: '0_3',
        parameters: {x1: 0.625, x2: 5.625},
        y: {metric_a: 18.111011269006838},
        y_hat: {metric_a: 18.210431712993607},
        se: {metric_a: 2.0},
        se_hat: {metric_a: 1.8146640929996618},
        context_stratum: null,
      },
      '0_30': {
        name: '0_30',
        parameters: {x1: -4.53125, x2: 7.96875},
        y: {metric_a: 70.60758289753784},
        y_hat: {metric_a: 69.88626691833579},
        se: {metric_a: 2.0},
        se_hat: {metric_a: 1.9194974577120703},
        context_stratum: null,
      },
      '0_31': {
        name: '0_31',
        parameters: {x1: -4.296875, x2: 3.984375},
        y: {metric_a: 132.44958166552067},
        y_hat: {metric_a: 130.6862637772996},
        se: {metric_a: 2.0},
        se_hat: {metric_a: 1.9419227274289987},
        context_stratum: null,
      },
      '0_32': {
        name: '0_32',
        parameters: {x1: 3.203125, x2: 11.484375},
        y: {metric_a: 86.10574699756219},
        y_hat: {metric_a: 87.29877055461004},
        se: {metric_a: 2.0},
        se_hat: {metric_a: 1.8984250787958818},
        context_stratum: null,
      },
      '0_33': {
        name: '0_33',
        parameters: {x1: 6.953125, x2: 0.234375},
        y: {metric_a: 18.419596839926022},
        y_hat: {metric_a: 18.13390309112501},
        se: {metric_a: 2.0},
        se_hat: {metric_a: 1.956195650635719},
        context_stratum: null,
      },
      '0_34': {
        name: '0_34',
        parameters: {x1: -0.546875, x2: 7.734375},
        y: {metric_a: 18.882901179508327},
        y_hat: {metric_a: 18.327379729554814},
        se: {metric_a: 2.0},
        se_hat: {metric_a: 1.8297621345247095},
        context_stratum: null,
      },
      '0_35': {
        name: '0_35',
        parameters: {x1: 1.328125, x2: 2.109375},
        y: {metric_a: 16.326252826011096},
        y_hat: {metric_a: 16.17614054510517},
        se: {metric_a: 2.0},
        se_hat: {metric_a: 1.6574259779625848},
        context_stratum: null,
      },
      '0_36': {
        name: '0_36',
        parameters: {x1: 8.828125, x2: 9.609375},
        y: {metric_a: 59.69046760890842},
        y_hat: {metric_a: 60.38405497187411},
        se: {metric_a: 2.0},
        se_hat: {metric_a: 1.7424714498793386},
        context_stratum: null,
      },
      '0_37': {
        name: '0_37',
        parameters: {x1: 5.078125, x2: 5.859375},
        y: {metric_a: 34.68741610290739},
        y_hat: {metric_a: 34.691850354221835},
        se: {metric_a: 2.0},
        se_hat: {metric_a: 1.656749910352912},
        context_stratum: null,
      },
      '0_38': {
        name: '0_38',
        parameters: {x1: -2.421875, x2: 13.359375},
        y: {metric_a: 10.325915083003967},
        y_hat: {metric_a: 10.620592335500412},
        se: {metric_a: 2.0},
        se_hat: {metric_a: 1.745307934819514},
        context_stratum: null,
      },
      '0_39': {
        name: '0_39',
        parameters: {x1: -1.484375, x2: 1.171875},
        y: {metric_a: 66.7077438958342},
        y_hat: {metric_a: 65.9021467817675},
        se: {metric_a: 2.0},
        se_hat: {metric_a: 1.879147718556558},
        context_stratum: null,
      },
      '0_4': {
        name: '0_4',
        parameters: {x1: 8.125, x2: 13.125},
        y: {metric_a: 140.32747319783857},
        y_hat: {metric_a: 139.68454943395102},
        se: {metric_a: 2.0},
        se_hat: {metric_a: 1.956014281604798},
        context_stratum: null,
      },
      '0_5': {
        name: '0_5',
        parameters: {x1: 4.375, x2: 1.875},
        y: {metric_a: 6.954951737223514},
        y_hat: {metric_a: 7.201748742878472},
        se: {metric_a: 2.0},
        se_hat: {metric_a: 1.9026785640195447},
        context_stratum: null,
      },
      '0_6': {
        name: '0_6',
        parameters: {x1: -3.125, x2: 9.375},
        y: {metric_a: 8.57972117932429},
        y_hat: {metric_a: 8.39349515187719},
        se: {metric_a: 2.0},
        se_hat: {metric_a: 1.8651110263807402},
        context_stratum: null,
      },
      '0_7': {
        name: '0_7',
        parameters: {x1: -2.1875, x2: 4.6875},
        y: {metric_a: 33.738344621143135},
        y_hat: {metric_a: 34.13698538141282},
        se: {metric_a: 2.0},
        se_hat: {metric_a: 1.7858916244733023},
        context_stratum: null,
      },
      '0_8': {
        name: '0_8',
        parameters: {x1: 5.3125, x2: 12.1875},
        y: {metric_a: 136.34953133391176},
        y_hat: {metric_a: 136.4830772271403},
        se: {metric_a: 2.0},
        se_hat: {metric_a: 1.7941945441185363},
        context_stratum: null,
      },
      '0_9': {
        name: '0_9',
        parameters: {x1: 9.0625, x2: 0.9375},
        y: {metric_a: 2.5808075578294103},
        y_hat: {metric_a: 2.735688351636938},
        se: {metric_a: 2.0},
        se_hat: {metric_a: 1.9763667003773213},
        context_stratum: null,
      },
    },
    out_of_sample: {},
    status_quo_name: null,
  };
  const arm_name_to_parameters = {
    '0_0': {x1: 2.5, x2: 7.5},
    '0_1': {x1: 6.25, x2: 3.75},
    '0_10': {x1: 1.5625, x2: 8.4375},
    '0_11': {x1: -0.3125, x2: 2.8125},
    '0_12': {x1: 7.1875, x2: 10.3125},
    '0_13': {x1: 3.4375, x2: 6.5625},
    '0_14': {x1: -4.0625, x2: 14.0625},
    '0_15': {x1: -3.59375, x2: 7.03125},
    '0_16': {x1: 3.90625, x2: 14.53125},
    '0_17': {x1: 7.65625, x2: 3.28125},
    '0_18': {x1: 0.15625, x2: 10.78125},
    '0_19': {x1: 2.03125, x2: 1.40625},
    '0_2': {x1: -1.25, x2: 11.25},
    '0_20': {x1: 9.53125, x2: 8.90625},
    '0_21': {x1: 5.78125, x2: 5.15625},
    '0_22': {x1: -1.71875, x2: 12.65625},
    '0_23': {x1: -2.65625, x2: 2.34375},
    '0_24': {x1: 4.84375, x2: 9.84375},
    '0_25': {x1: 8.59375, x2: 6.09375},
    '0_26': {x1: 1.09375, x2: 13.59375},
    '0_27': {x1: -0.78125, x2: 4.21875},
    '0_28': {x1: 6.71875, x2: 11.71875},
    '0_29': {x1: 2.96875, x2: 0.46875},
    '0_3': {x1: 0.625, x2: 5.625},
    '0_30': {x1: -4.53125, x2: 7.96875},
    '0_31': {x1: -4.296875, x2: 3.984375},
    '0_32': {x1: 3.203125, x2: 11.484375},
    '0_33': {x1: 6.953125, x2: 0.234375},
    '0_34': {x1: -0.546875, x2: 7.734375},
    '0_35': {x1: 1.328125, x2: 2.109375},
    '0_36': {x1: 8.828125, x2: 9.609375},
    '0_37': {x1: 5.078125, x2: 5.859375},
    '0_38': {x1: -2.421875, x2: 13.359375},
    '0_39': {x1: -1.484375, x2: 1.171875},
    '0_4': {x1: 8.125, x2: 13.125},
    '0_5': {x1: 4.375, x2: 1.875},
    '0_6': {x1: -3.125, x2: 9.375},
    '0_7': {x1: -2.1875, x2: 4.6875},
    '0_8': {x1: 5.3125, x2: 12.1875},
    '0_9': {x1: 9.0625, x2: 0.9375},
  };
  const f = [
    97.998646581203,
    85.09026951171091,
    71.38978520965746,
    57.569259826682966,
    44.4317019994308,
    32.76002697792867,
    23.139216578416786,
    15.87278896472938,
    11.019425380518676,
    8.44270479692856,
    7.851057503654111,
    8.837744826718698,
    10.926302551647986,
    13.620979939316442,
    16.45901244505896,
    19.062473966598326,
    21.184292629938707,
    22.71532273100996,
    23.65218304537617,
    24.06566606456792,
    24.078231295962095,
    23.849474442889367,
    23.566035367576628,
    23.431238781341232,
    23.650871760961007,
    24.41569673967271,
    25.882833397295336,
    28.148509450022388,
    31.216892153987096,
    34.98295144252011,
    39.238558195625785,
    43.69992801109773,
    48.04495100560361,
    51.951816136249114,
    55.133515941615556,
    57.36486906727509,
    58.5008433345729,
    58.48532833452875,
    57.349463535641185,
    55.200659515540316,
    52.20563609649514,
    48.57117098923302,
    44.525401444699725,
    40.300959873234405,
    36.11956120979852,
    32.177037327164484,
    28.628646975195423,
    25.577878437786612,
    23.075713458837175,
    21.13104679893421,
  ];
  const fit_data = [
    {
      metric_name: 'metric_a',
      arm_name: '0_0',
      mean: 24.129964413622268,
      sem: 2.0,
    },
    {
      metric_name: 'metric_a',
      arm_name: '0_1',
      mean: 26.624171220014905,
      sem: 2.0,
    },
    {
      metric_name: 'metric_a',
      arm_name: '0_10',
      mean: 31.321658517480987,
      sem: 2.0,
    },
    {
      metric_name: 'metric_a',
      arm_name: '0_11',
      mean: 32.8083830521407,
      sem: 2.0,
    },
    {
      metric_name: 'metric_a',
      arm_name: '0_12',
      mean: 98.34760790686913,
      sem: 2.0,
    },
    {
      metric_name: 'metric_a',
      arm_name: '0_13',
      mean: 21.127854002025494,
      sem: 2.0,
    },
    {
      metric_name: 'metric_a',
      arm_name: '0_14',
      mean: 4.47623958200588,
      sem: 2.0,
    },
    {
      metric_name: 'metric_a',
      arm_name: '0_15',
      mean: 41.77178986158392,
      sem: 2.0,
    },
    {
      metric_name: 'metric_a',
      arm_name: '0_16',
      mean: 166.3236985012143,
      sem: 2.0,
    },
    {
      metric_name: 'metric_a',
      arm_name: '0_17',
      mean: 15.473497571024838,
      sem: 2.0,
    },
    {
      metric_name: 'metric_a',
      arm_name: '0_18',
      mean: 44.753611366285824,
      sem: 2.0,
    },
    {
      metric_name: 'metric_a',
      arm_name: '0_19',
      mean: 9.320218587634006,
      sem: 2.0,
    },
    {
      metric_name: 'metric_a',
      arm_name: '0_2',
      mean: 22.383482484999874,
      sem: 2.0,
    },
    {
      metric_name: 'metric_a',
      arm_name: '0_20',
      mean: 40.64753379393727,
      sem: 2.0,
    },
    {
      metric_name: 'metric_a',
      arm_name: '0_21',
      mean: 34.736739987996074,
      sem: 2.0,
    },
    {
      metric_name: 'metric_a',
      arm_name: '0_22',
      mean: 21.110094161886266,
      sem: 2.0,
    },
    {
      metric_name: 'metric_a',
      arm_name: '0_23',
      mean: 78.863835999175,
      sem: 2.0,
    },
    {
      metric_name: 'metric_a',
      arm_name: '0_24',
      mean: 83.88052752898044,
      sem: 2.0,
    },
    {
      metric_name: 'metric_a',
      arm_name: '0_25',
      mean: 21.424386764652148,
      sem: 2.0,
    },
    {
      metric_name: 'metric_a',
      arm_name: '0_26',
      mean: 98.68064045456067,
      sem: 2.0,
    },
    {
      metric_name: 'metric_a',
      arm_name: '0_27',
      mean: 26.449512503021804,
      sem: 2.0,
    },
    {
      metric_name: 'metric_a',
      arm_name: '0_28',
      mean: 130.64996148022834,
      sem: 2.0,
    },
    {
      metric_name: 'metric_a',
      arm_name: '0_29',
      mean: 4.323605035294334,
      sem: 2.0,
    },
    {
      metric_name: 'metric_a',
      arm_name: '0_3',
      mean: 18.111011269006838,
      sem: 2.0,
    },
    {
      metric_name: 'metric_a',
      arm_name: '0_30',
      mean: 70.60758289753784,
      sem: 2.0,
    },
    {
      metric_name: 'metric_a',
      arm_name: '0_31',
      mean: 132.44958166552067,
      sem: 2.0,
    },
    {
      metric_name: 'metric_a',
      arm_name: '0_32',
      mean: 86.10574699756219,
      sem: 2.0,
    },
    {
      metric_name: 'metric_a',
      arm_name: '0_33',
      mean: 18.419596839926022,
      sem: 2.0,
    },
    {
      metric_name: 'metric_a',
      arm_name: '0_34',
      mean: 18.882901179508327,
      sem: 2.0,
    },
    {
      metric_name: 'metric_a',
      arm_name: '0_35',
      mean: 16.326252826011096,
      sem: 2.0,
    },
    {
      metric_name: 'metric_a',
      arm_name: '0_36',
      mean: 59.69046760890842,
      sem: 2.0,
    },
    {
      metric_name: 'metric_a',
      arm_name: '0_37',
      mean: 34.68741610290739,
      sem: 2.0,
    },
    {
      metric_name: 'metric_a',
      arm_name: '0_38',
      mean: 10.325915083003967,
      sem: 2.0,
    },
    {
      metric_name: 'metric_a',
      arm_name: '0_39',
      mean: 66.7077438958342,
      sem: 2.0,
    },
    {
      metric_name: 'metric_a',
      arm_name: '0_4',
      mean: 140.32747319783857,
      sem: 2.0,
    },
    {
      metric_name: 'metric_a',
      arm_name: '0_5',
      mean: 6.954951737223514,
      sem: 2.0,
    },
    {
      metric_name: 'metric_a',
      arm_name: '0_6',
      mean: 8.57972117932429,
      sem: 2.0,
    },
    {
      metric_name: 'metric_a',
      arm_name: '0_7',
      mean: 33.738344621143135,
      sem: 2.0,
    },
    {
      metric_name: 'metric_a',
      arm_name: '0_8',
      mean: 136.34953133391176,
      sem: 2.0,
    },
    {
      metric_name: 'metric_a',
      arm_name: '0_9',
      mean: 2.5808075578294103,
      sem: 2.0,
    },
  ];
  const grid = [
    -5.0,
    -4.6938775510204085,
    -4.387755102040816,
    -4.081632653061225,
    -3.7755102040816326,
    -3.4693877551020407,
    -3.163265306122449,
    -2.857142857142857,
    -2.5510204081632653,
    -2.2448979591836733,
    -1.9387755102040813,
    -1.6326530612244898,
    -1.3265306122448979,
    -1.020408163265306,
    -0.7142857142857144,
    -0.408163265306122,
    -0.1020408163265305,
    0.204081632653061,
    0.5102040816326534,
    0.8163265306122449,
    1.1224489795918373,
    1.4285714285714288,
    1.7346938775510203,
    2.0408163265306127,
    2.3469387755102042,
    2.6530612244897958,
    2.959183673469388,
    3.2653061224489797,
    3.571428571428571,
    3.8775510204081627,
    4.183673469387756,
    4.4897959183673475,
    4.795918367346939,
    5.1020408163265305,
    5.408163265306122,
    5.714285714285715,
    6.020408163265307,
    6.326530612244898,
    6.63265306122449,
    6.938775510204081,
    7.244897959183675,
    7.551020408163266,
    7.857142857142858,
    8.16326530612245,
    8.46938775510204,
    8.775510204081632,
    9.081632653061225,
    9.387755102040817,
    9.693877551020408,
    10.0,
  ];
  const metric = 'metric_a';
  const param = 'x1';
  const rel = false;
  const setx = {x1: 2.353515625, x2: 7.5};
  const sd = [
    3.6982014075048477,
    2.3869162162409805,
    1.7710791120396967,
    1.654109770748382,
    1.6479146028668255,
    1.7390022884399634,
    2.0555837784162243,
    2.5294338121416384,
    2.994627588908216,
    3.3219339353438335,
    3.433998517489504,
    3.2976570754646066,
    2.929356093432033,
    2.4183995877483433,
    1.963918831588927,
    1.8266017394118879,
    2.0019627093304955,
    2.217730861943818,
    2.3087743941981644,
    2.2598904708904404,
    2.1280315491272304,
    1.985662546062033,
    1.8684742158048768,
    1.7549214621586373,
    1.6504998716989991,
    1.6455791686254602,
    1.781001057675659,
    1.999824504979558,
    2.2477073284674307,
    2.458667332105268,
    2.556718799060391,
    2.5318033716526336,
    2.487598706262191,
    2.5875456613320806,
    2.895454474759736,
    3.319747941872684,
    3.7330047743831694,
    4.051182299644566,
    4.2341948020786475,
    4.268545913676225,
    4.151746942487063,
    3.8840640936220665,
    3.4769772910729464,
    2.9807267245913915,
    2.5154478302517598,
    2.2626134278214693,
    2.364023921526548,
    2.8550868432380776,
    3.7586056088371502,
    5.092047485789122,
  ];
  const is_log = false;

  // format data
  const res = relativize_data(f, sd, rel, arm_data, metric);
  const f_final = res[0];
  const sd_final = res[1];

  // get data for standard deviation fill plot
  const sd_upper = [];
  const sd_lower = [];
  for (let i = 0; i < sd.length; i++) {
    sd_upper.push(f_final[i] + 2 * sd_final[i]);
    sd_lower.push(f_final[i] - 2 * sd_final[i]);
  }
  const grid_rev = copy_and_reverse(grid);
  const sd_lower_rev = copy_and_reverse(sd_lower);
  const sd_x = grid.concat(grid_rev);
  const sd_y = sd_upper.concat(sd_lower_rev);

  // get data for observed arms and error bars
  const arm_x = [];
  const arm_y = [];
  const arm_sem = [];
  fit_data.forEach(row => {
    parameters = arm_name_to_parameters[row['arm_name']];
    plot = true;
    Object.keys(setx).forEach(p => {
      if (p !== param && parameters[p] !== setx[p]) {
        plot = false;
      }
    });
    if (plot === true) {
      arm_x.push(parameters[param]);
      arm_y.push(row['mean']);
      arm_sem.push(row['sem']);
    }
  });

  const arm_res = relativize_data(arm_y, arm_sem, rel, arm_data, metric);
  const arm_y_final = arm_res[0];
  const arm_sem_final = arm_res[1].map(x => x * 2);

  // create traces
  const f_trace = {
    x: grid,
    y: f_final,
    showlegend: false,
    hoverinfo: 'x+y',
    line: {
      color: 'rgba(128, 177, 211, 1)',
    },
  };

  const arms_trace = {
    x: arm_x,
    y: arm_y_final,
    mode: 'markers',
    error_y: {
      type: 'data',
      array: arm_sem_final,
      visible: true,
      color: 'black',
    },
    line: {
      color: 'black',
    },
    showlegend: false,
    hoverinfo: 'x+y',
  };

  const sd_trace = {
    x: sd_x,
    y: sd_y,
    fill: 'toself',
    fillcolor: 'rgba(128, 177, 211, 0.2)',
    line: {
      color: 'transparent',
    },
    showlegend: false,
    hoverinfo: 'none',
  };

  traces = [sd_trace, f_trace, arms_trace];

  // iterate over out-of-sample arms
  let i = 1;
  Object.keys(arm_data['out_of_sample']).forEach(generator_run_name => {
    const ax = [];
    const ay = [];
    const asem = [];
    const atext = [];

    Object.keys(arm_data['out_of_sample'][generator_run_name]).forEach(
      arm_name => {
        const parameters =
          arm_data['out_of_sample'][generator_run_name][arm_name]['parameters'];
        plot = true;
        Object.keys(setx).forEach(p => {
          if (p !== param && parameters[p] !== setx[p]) {
            plot = false;
          }
        });
        if (plot === true) {
          ax.push(parameters[param]);
          ay.push(
            arm_data['out_of_sample'][generator_run_name][arm_name]['y_hat'][
              metric
            ],
          );
          asem.push(
            arm_data['out_of_sample'][generator_run_name][arm_name]['se_hat'][
              metric
            ],
          );
          atext.push('<em>Candidate ' + arm_name + '</em>');
        }
      },
    );

    const out_of_sample_arm_res = relativize_data(
      ay,
      asem,
      rel,
      arm_data,
      metric,
    );
    const ay_final = out_of_sample_arm_res[0];
    const asem_final = out_of_sample_arm_res[1].map(x => x * 2);

    traces.push({
      hoverinfo: 'text',
      legendgroup: generator_run_name,
      marker: {color: 'black', symbol: i, opacity: 0.5},
      mode: 'markers',
      error_y: {
        type: 'data',
        array: asem_final,
        visible: true,
        color: 'black',
      },
      name: generator_run_name,
      text: atext,
      type: 'scatter',
      xaxis: 'x',
      x: ax,
      yaxis: 'y',
      y: ay_final,
    });

    i += 1;
  });

  // layout
  const xrange = axis_range(grid, is_log);
  const xtype = is_log ? 'log' : 'linear';

  layout = {
    hovermode: 'closest',
    xaxis: {
      anchor: 'y',
      autorange: false,
      exponentformat: 'e',
      range: xrange,
      tickfont: {size: 11},
      tickmode: 'auto',
      title: param,
      type: xtype,
    },
    yaxis: {
      anchor: 'x',
      tickfont: {size: 11},
      tickmode: 'auto',
      title: metric,
    },
  };
  return {
    layout: layout,
    traces: traces,
  };
}

const slicePlotData = getPlotData();

Plotly.newPlot('slice', slicePlotData['traces'], slicePlotData['layout'], {
  responsive: true,
  showLink: false,
});
