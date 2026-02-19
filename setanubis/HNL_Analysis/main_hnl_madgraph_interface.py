from SetAnubis.core.MadGraph.adapters.input.GeneralCardInterface import GeneralCardInterface, MadGraphCommandConfig
from SetAnubis.core.MadGraph.adapters.input.MadGraphInterface import MadgraphInterface
from SetAnubis.core.interfaces import SetAnubisInterface

from SetAnubis.core.MadGraph.domain.CommandSectionType import CommandSectionType    # Added so that I could explicitly define what ell and vv was in the jobcard for the muon-coupled HNL production

from SetAnubis.core.MadGraph.adapters.output.MadGraphLocalRunner import MadGraphLocalRunner

if __name__ == "__main__":
    
    dry_run = False
    
    neo = SetAnubisInterface("/usera/fs568/set-anubis/Assets/UFO/SM_HeavyN_CKM_AllMasses_LO")

    config = MadGraphCommandConfig(
        neo_set_anubis=neo,
        cache=False,
        model_in_madgraph="SM_HeavyN_CKM_AllMasses_LO",
        shower="py8",
        madspin="ON")
    
    card_interface = GeneralCardInterface(config)
    
    param_card = card_interface.param_card
        
    runcard_editor = card_interface.run_card_builder
    runcard_editor.set("nevents", 2000)
    runcard_str = runcard_editor.serialize()

    builder_madspin = card_interface.madspin_builder
    builder_madspin.add_decay("decay n1 > ell ell vv")
    madspin_str = builder_madspin.serialize()
    
    pythia_str = card_interface.pythia_builder.serialize()
    
    
    jobcard = card_interface.josbscript_builder
    # Redefine ell as muons for muon-coupled HNL production
    jobcard.add_special_section(CommandSectionType.DEFINITIONS, "define ell = mu+ mu-")
    jobcard.add_special_section(CommandSectionType.DEFINITIONS, "define vv = vm vm~")
    jobcard.add_process("generate p p > n1 ell # [QCD]")
    # jobcard.add_special_section(CommandSectionType.SETTINGS,"set nb_core -1")
    jobcard.set_output_launch("HNL_Condor_CCDY_qqmu_Scan_1")
    jobcard.configure_cards()
    jobcard.add_auto_width("WN1")
    jobcard.add_parameter_scan("VeN1", "[0.0]")
    jobcard.add_parameter_scan("VmuN1", "[0.00316,0.001,0.000316]")
    jobcard.add_parameter_scan("mN1", "[0.8,1.0,3,5,7]")
    jobscript_str = jobcard.serialize()

    print("------------------------------------------------------------------------------------------")
    print(jobscript_str)
    print("------------------------------------------------------------------------------------------")
    print(madspin_str)
    print("------------------------------------------------------------------------------------------")
    print(pythia_str)
    print("------------------------------------------------------------------------------------------")
    
    print(runcard_str)
    print("------------------------------------------------------------------------------------------")
    
    print(param_card)
    print("------------------------------------------------------------------------------------------")
    
    mlr = MadGraphLocalRunner()
    
    #Error if madgraph is not installed but that's okay
    mg = MadgraphInterface(
        madgraph_runner=mlr,
        jobscript_str=jobscript_str,
        param_card_str=param_card,
        run_card_str=runcard_str,
        pythia_card_str=pythia_str,
        madspin_card_str=madspin_str
    )

    if not dry_run:
        mg.run()
        mg.retrieve_events()
