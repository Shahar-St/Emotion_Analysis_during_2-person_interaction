tasks_to_features_dict = {
    'FET': [
        'ExpectancySelection_neg',
        'ExpectancySelection_pos',
        'ExpectancyRT_negative.accept',
        'ExpectancyRT_negative.reject',
        'ExpectancyRT_positive.accept',
        'ExpectancyRT_positive.reject',
        'ExpectancyConfidance_positive.accept',
        'ExpectancyConfidance_positive.reject',
    ],
    'WSAP': [
        'Interpretation_NegSelection', 'InterpretationRT_Neg', 'InterpretationRT_Pos', 'Interpretation_Rtneg-pos'
    ],
    'FAFT': [
        'INTERcongruent.neg', 'INTERcongruent.neutral', 'INTERcongruent.pos', 'INTERincongruent.neg',
        'INTERincongruent.neutral', 'INTERincongruent.pos', 'InterferenceCong_Neg-Neu', 'InterferenceCong_Pos-Neu',
        'InterferenceIncong_Pos-Neu',
        'InterferenceIncong_Neg-Neu'
    ],
    'WIT': [
        'MEMPriming_neg', 'MEMPrimig_neut', 'MEMPriming_pos', 'MemoryImplicit_neg-neu', 'MemoryImplicit_pos-neu',
        'ExpMemNegative', 'ExpMemPositive', 'ExpMemNeutral', 'EXPMEM_Neg-Neu', 'EXPMEM_Pos-Neu'
    ],
    'EDPT': [
        'DP.50.Angry.Congruent', 'DP.50.Angry.Incongruent', 'DP.50.Happy.Congruent', 'DP.50.Happy.Incongruent',
        'DP.50.Sad.Congruent', 'DP.50.Sad.Incongruent', 'DP.1000.Angry.Congruent', 'DP.1000.Angry.Incongruent',
        'DP.1000.Happy.Congruent', 'DP.1000.Happy.Incongruent', 'DP.1000.Sad.Congruent', 'DP.1000.Sad.Incongruent',
        'DPCongruency_happy50', 'DPCongruency_happy1000', 'DPCongruency_angry50', 'DPCongruency_angry1000',
        'DPCongruency_sad50', 'DPCongruency_sad1000'
    ],
    'ISC': [
        'ISC_Neutral', 'ISC_Emotional', 'ISC_EMO-NEU', 'ISC_N-N', 'ISC_E-N', 'ISC_N-E', 'ISC_E-E'
    ]
}

implicit_features = [
    'ISC_Neutral',
    'ISC_Emotional',
    'ISC_EMO-NEU',
    'ISC_N-N',
    'ISC_E-N',
    'ISC_N-E',
    'ISC_E-E',
    'DP.50.Angry.Congruent',
    'DP.50.Angry.Incongruent',
    'DP.50.Happy.Congruent',
    'DP.50.Happy.Incongruent',
    'DP.50.Sad.Congruent',
    'DP.50.Sad.Incongruent',
    'DPCongruency_happy50',
    'DPCongruency_angry50',
    'DPCongruency_sad50',
    'MEMPriming_neg',
    'MEMPrimig_neut',
    'MEMPriming_pos',
    'MemoryImplicit_neg-neu',
    'MemoryImplicit_pos-neu',
    'INTERcongruent.neg',
    'INTERcongruent.neutral',
    'INTERcongruent.pos',
    'INTERincongruent.neg',
    'INTERincongruent.neutral',
    'INTERincongruent.pos',
    'InterferenceCong_Neg-Neu',
    'InterferenceCong_Pos-Neu',
    'InterferenceIncong_Pos-Neu',
    'InterferenceIncong_Neg-Neu',
    'InterpretationRT_Neg',
    'InterpretationRT_Pos',
    'Interpretation_Rtneg-pos',
    'ExpectancyRT_negative.accept',
    'ExpectancyRT_negative.reject',
    'ExpectancyRT_positive.accept',
    'ExpectancyRT_positive.reject',
    'ExpectancyRejection',
    'ExpectancyAcceptance'
]

explicit_features = [
    'DP.1000.Angry.Congruent',
    'DP.1000.Angry.Incongruent',
    'DP.1000.Happy.Congruent',
    'DP.1000.Happy.Incongruent',
    'DP.1000.Sad.Congruent',
    'DP.1000.Sad.Incongruent',
    'DPCongruency_happy1000',
    'DPCongruency_angry1000',
    'DPCongruency_sad1000',
    'ExpMemNegative',
    'ExpMemPositive',
    'ExpMemNeutral',
    'EXPMEM_Neg-Neu',
    'EXPMEM_Pos-Neu',
    'Interpretation_NegSelection',
    'ExpectancySelection_neg',
    'ExpectancySelection_pos',
    'ExpectancyConfidance_negative.accept',
    'ExpectancyConfidance_negative.reject',
    'ExpectancyConfidance_positive.accept',
    'ExpectancyConfidance_positive.reject'
]

thresholds = {
    'STAI': 50,
    'BDI': 29
}
