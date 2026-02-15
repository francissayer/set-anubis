from SetAnubis.core.BranchingRatio.adapters.input.DecayInterface import DecayInterface, CalculationDecayStrategy
from SetAnubis.core.ModelCore.adapters.input.SetAnubisInteface import SetAnubisInterface
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PY_SCRIPT_PATH = os.path.join(CURRENT_DIR, "Width_Calculator.py")

if __name__ == "__main__":
    neosetanubis = SetAnubisInterface("/usera/fs568/set-anubis/Assets/UFO/ALP_linear_UFO_WIDTH")
    
    neosetanubis.set_leaf_param("ZERO", 0)
    all_particles = neosetanubis.get_all_particles()
    all_params = neosetanubis.get_all_parameters()
    
    br = DecayInterface(neosetanubis)
    
    alp_decay_list = [
        {"mother": 9000005, "daughters": [22, 22]},     # ax -> γγ
        {"mother": 9000005, "daughters": [22, 23]},     # ax -> γZ
        {"mother": 9000005, "daughters": [5, -5]},      # ax -> bb̄
        {"mother": 9000005, "daughters": [4, -4]},      # ax -> cc̄
        {"mother": 9000005, "daughters": [1, -1]},      # ax -> dd̄
        {"mother": 9000005, "daughters": [11, -11]},    # ax -> e⁻e⁺
        {"mother": 9000005, "daughters": [21, 21]},     # ax -> gg
        {"mother": 9000005, "daughters": [13, -13]},    # ax -> μ⁻μ⁺
        {"mother": 9000005, "daughters": [3, -3]},      # ax -> ss̄
        {"mother": 9000005, "daughters": [6, -6]},      # ax -> tt̄     
        {"mother": 9000005, "daughters": [15, -15]},    # ax -> τ⁻τ⁺
        {"mother": 9000005, "daughters": [2, -2]},      # ax -> uū   
        {"mother": 9000005, "daughters": [-24, 24]},    # ax -> W⁻W⁺
        {"mother": 9000005, "daughters": [23, 23]},     # ax -> ZZ
    ]
    
    br.add_decays(alp_decay_list, CalculationDecayStrategy.PYTHON, config={"script_path" : PY_SCRIPT_PATH})
    
    gamma_aa = br.get_decay(9000005, [22, 22]).real
    gamma_aZ = br.get_decay(9000005, [22, 23]).real
    gamma_bb = br.get_decay(9000005, [5, -5]).real
    gamma_cc = br.get_decay(9000005, [4, -4]).real
    gamma_dd = br.get_decay(9000005, [1, -1]).real
    gamma_ee = br.get_decay(9000005, [11, -11]).real
    gamma_gg = br.get_decay(9000005, [21, 21]).real
    gamma_mumu = br.get_decay(9000005, [13, -13]).real
    gamma_ss = br.get_decay(9000005, [3, -3]).real
    gamma_tt = br.get_decay(9000005, [6, -6]).real
    gamma_tautau = br.get_decay(9000005, [15, -15]).real
    gamma_uu = br.get_decay(9000005, [2, -2]).real
    gamma_WW = br.get_decay(9000005, [-24, 24]).real
    gamma_ZZ = br.get_decay(9000005, [23, 23]).real
    # .real to convert from complex format to floats
    
    print(f"Gamma(ax->γγ) = {gamma_aa}")
    print(f"Gamma(ax->γZ) = {gamma_aZ}")
    print(f"Gamma(ax->bb̄) = {gamma_bb}")
    print(f"Gamma(ax->cc̄) = {gamma_cc}")
    print(f"Gamma(ax->dd̄) = {gamma_dd}")
    print(f"Gamma(ax->e⁻e⁺) = {gamma_ee}")
    print(f"Gamma(ax->gg) = {gamma_gg}")
    print(f"Gamma(ax->μ⁻μ⁺) = {gamma_mumu}")
    print(f"Gamma(ax->ss̄) = {gamma_ss}")
    print(f"Gamma(ax->tt̄) = {gamma_tt}")
    print(f"Gamma(ax->τ⁻τ⁺) = {gamma_tautau}")
    print(f"Gamma(ax->uū) = {gamma_uu}")
    print(f"Gamma(ax->W⁻W⁺) = {gamma_WW}")
    print(f"Gamma(ax->ZZ) = {gamma_ZZ}")
    
    total_width = sum([
        gamma_aa, gamma_aZ, gamma_bb, gamma_cc, gamma_dd,
        gamma_ee, gamma_gg, gamma_mumu, gamma_ss, gamma_tt,
        gamma_tautau, gamma_uu, gamma_WW, gamma_ZZ
    ])
    print(f"Total width Gamma(ax) = {total_width}")
    
    #total_width = br.get_decay_tot(9000005)
    #print(f"Total width Gamma(ax) = {total_width}")

    #brs = br.get_brs(9000005)
    #for item in brs:
    #    print(item)
