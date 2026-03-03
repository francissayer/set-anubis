from SetAnubis.core.MadGraph.adapters.input.GeneralCardInterface import GeneralCardInterface, MadGraphCommandConfig
from SetAnubis.core.MadGraph.adapters.input.MadGraphInterface import MadgraphInterface
from SetAnubis.core.interfaces import SetAnubisInterface

from SetAnubis.core.MadGraph.adapters.output.MadGraphLocalRunner import MadGraphLocalRunner

if __name__ == "__main__":
    
    for i in range(1,13):

        """
        Paramater use to choose if we want to produce the card only (True) or run madgraph on docker.
        """
        dry_run = False # True just checks that the script runs (i.e. just prints the cards, no runs), set to False to actually generate events
        
        """
        General interface of the neo-set-anubis pipeline. Need the path to the UFO as an input.
        
        Everything concerning this interface is available in the ModelCore.exampleNeoSetAnubisInterface.py example.
        """
        neo = SetAnubisInterface("/usera/fs568/set-anubis/Assets/UFO/ALP_linear_UFO_WIDTH")
        
        """
        Configuration for the MagraphInterface (for writing cards). Few inputs are needed :
        
            -   neo_set_anubis : General NeoSetAnubis interface, with the ufo_path and all the particles/parameters.
            -   cards_path  :   Path to the cards in the docker container. No need to change it it will only break things (or be sure of what you're doing !).
            -   cache : If we want to use what's already in MadGraph, Generally put it to False only you're sure or doing the same scan than before.
            -   model_in_madgraph : name of the UFO, used in madgraph to import the model (from Feynrule).
            -   shower  :   shower option in madgraph, tell it which software will deal with the shower (use pythia by default).
            -   madspin :   madspin option, whether to use it for the decay of the LLP or not.
        """
        config = MadGraphCommandConfig(
            neo_set_anubis=neo,
            cache=False,
            model_in_madgraph="ALP_linear_UFO_WIDTH",
            shower="py8",
            madspin="ON")
        
        """
        General interface for the cards creation. Everything on the heap, no file writing or anything.
        
        The three main cards are the runcard, param_card and the jobcard.
        
        param_card : The param_card is automatically created by the UFO (writing part) and will use the default value of the parameters. In order to change a parameters value, either change it in the UFO or 
        in the jobscard (parameter scan can be used with one value to set the parameters's value).
        
        run_card : The run_card can be edited to change the number of events, the parton distribution function, the energy of the beam, some cuts or other general parameters.
        
        jobcard : The jobcard is used to select the differents process to generate the LLP, and choose the parameters for the scan. See below the example.
        
        Two other cards are used for madspin and pythia : 
        
        pythia_card : The pythia_card is automatically generated and shouldn't be changed
        """
        card_interface = GeneralCardInterface(config)
        
        param_card = card_interface.param_card

        runcard_editor = card_interface.run_card_builder
        runcard_editor.set("nevents", 2000)
        #############################################################################################
        # # I've added this to use LHAPDF for PDFs in MadGraph (change ID to desired set)
        # runcard_editor.set("pdlabel", "lhapdf")
        # runcard_editor.set("lhaid", 331100)  # example: replace 306000 with your LHAPDF numeric ID
        #############################################################################################
        runcard_str = runcard_editor.serialize()

        builder_madspin = card_interface.madspin_builder
        # Include all possible final states - but don't include kinematically forbidden decays as MadSpin will error out
        #builder_madspin.add_decay("decay ax > e- e+")       # 2m_e = 1.022e-3 GeV
        #builder_madspin.add_decay("decay ax > mu- mu+")     # 2m_mu = 0.21132 GeV
        #builder_madspin.add_decay("decay ax > ta- ta+")     # 2m_tau = 3.554 GeV
        #builder_madspin.add_decay("decay ax > u u~")        # 2m_u = 5.1e-3 GeV
        #builder_madspin.add_decay("decay ax > d d~")        # 2m_d = 1.008e-2 GeV
        #builder_madspin.add_decay("decay ax > s s~")        # 2m_s = 0.202 GeV
        #builder_madspin.add_decay("decay ax > c c~")        # 2m_c = 2.54 GeV
        #builder_madspin.add_decay("decay ax > b b~")        # 2m_b = 9.4 GeV
        #builder_madspin.add_decay("decay ax > t t~")        # 2m_t = 346.6 GeV
        builder_madspin.add_decay("decay ax > ll ll")
        builder_madspin.add_decay("decay ax > qq qq") 
        madspin_str = builder_madspin.serialize()
        
        pythia_str = card_interface.pythia_builder.serialize()
        #############################################################################################
        # # I've added this to ensure Pythia uses the same LHAPDF set (prepend these lines if not present)
        # pythia_pdf_header = "PDF:useLHAPDF = on\nPDF:LHAPDFset = NNPDF40_nnlo_as_01180\n"
        # if 'PDF:useLHAPDF' not in pythia_str:
        #     pythia_str = pythia_pdf_header + pythia_str
        #############################################################################################

        
        jobcard = card_interface.josbscript_builder
        jobcard.add_process("define ll = e- e+ mu- mu+ ta- ta+ \n define qq = u u~ d d~ s s~ c c~ b b~ t t~ \n generate p p > ax W-")
        if i == 1:
            jobcard.set_output_launch("ALP_axW-_scan_1")
            jobcard.configure_cards()
            jobcard.add_auto_width("WALP")  # Automatically compute ALP width from decay formulas
            jobcard.add_parameter_scan("Ma", "[0.0562]")      # ALP mass in GeV
            jobcard.add_parameter_scan("fa", "[1000]")     # ALP decay constant in GeV
            jobcard.add_parameter_scan("CaPhi", "[1,0.316,0.1,0.0316,0.01,0.00316]")   # Universal ALP-fermion coupling
        elif i == 2:
            jobcard.set_output_launch("ALP_axW-_scan_2")
            jobcard.configure_cards()
            jobcard.add_auto_width("WALP")  # Automatically compute ALP width from decay formulas
            jobcard.add_parameter_scan("Ma", "[0.1]")      # ALP mass in GeV
            jobcard.add_parameter_scan("fa", "[1000]")     # ALP decay constant in GeV
            jobcard.add_parameter_scan("CaPhi", "[1,0.316,0.1,0.0316,0.01,0.00316,0.001,0.000316]")   # Universal ALP-fermion coupling
        elif i == 3:
            jobcard.set_output_launch("ALP_axW-_scan_3")
            jobcard.configure_cards()
            jobcard.add_auto_width("WALP")  # Automatically compute ALP width from decay formulas
            jobcard.add_parameter_scan("Ma", "[0.178]")      # ALP mass in GeV
            jobcard.add_parameter_scan("fa", "[1000]")     # ALP decay constant in GeV
            jobcard.add_parameter_scan("CaPhi", "[1,0.316,0.1,0.0316,0.01,0.00316,0.001,0.000316,0.0001]")   # Universal ALP-fermion coupling
        elif i == 4:
            jobcard.set_output_launch("ALP_axW-_scan_4")
            jobcard.configure_cards()
            jobcard.add_auto_width("WALP")  # Automatically compute ALP width from decay formulas
            jobcard.add_parameter_scan("Ma", "[0.316]")      # ALP mass in GeV
            jobcard.add_parameter_scan("fa", "[1000]")     # ALP decay constant in GeV
            jobcard.add_parameter_scan("CaPhi", "[0.316,0.1,0.0316,0.01,0.00316,0.001,0.000316,0.0001,0.0000316]")   # Universal ALP-fermion coupling
        elif i == 5:
            jobcard.set_output_launch("ALP_axW-_scan_5")
            jobcard.configure_cards()
            jobcard.add_auto_width("WALP")  # Automatically compute ALP width from decay formulas
            jobcard.add_parameter_scan("Ma", "[0.562]")      # ALP mass in GeV
            jobcard.add_parameter_scan("fa", "[1000]")     # ALP decay constant in GeV
            jobcard.add_parameter_scan("CaPhi", "[0.1,0.0316,0.01,0.00316,0.001,0.000316,0.0001,0.0000316,0.00001]")   # Universal ALP-fermion coupling
        elif i == 6:
            jobcard.set_output_launch("ALP_axW-_scan_6")
            jobcard.configure_cards()
            jobcard.add_auto_width("WALP")  # Automatically compute ALP width from decay formulas
            jobcard.add_parameter_scan("Ma", "[1]")      # ALP mass in GeV
            jobcard.add_parameter_scan("fa", "[1000]")     # ALP decay constant in GeV
            jobcard.add_parameter_scan("CaPhi", "[0.01,0.00316,0.001,0.000316,0.0001,0.0000316,0.00001,0.00000316]")   # Universal ALP-fermion coupling
        elif i == 7:
            jobcard.set_output_launch("ALP_axW-_scan_7")
            jobcard.configure_cards()
            jobcard.add_auto_width("WALP")  # Automatically compute ALP width from decay formulas
            jobcard.add_parameter_scan("Ma", "[1.78]")      # ALP mass in GeV
            jobcard.add_parameter_scan("fa", "[1000]")     # ALP decay constant in GeV
            jobcard.add_parameter_scan("CaPhi", "[0.00316,0.001,0.000316,0.0001,0.0000316,0.00001,0.00000316,0.000001]")   # Universal ALP-fermion coupling
        elif i == 8:
            jobcard.set_output_launch("ALP_axW-_scan_8")
            jobcard.configure_cards()
            jobcard.add_auto_width("WALP")  # Automatically compute ALP width from decay formulas
            jobcard.add_parameter_scan("Ma", "[3.16]")      # ALP mass in GeV
            jobcard.add_parameter_scan("fa", "[1000]")     # ALP decay constant in GeV
            jobcard.add_parameter_scan("CaPhi", "[0.001,0.000316,0.0001,0.0000316,0.00001,0.00000316,0.000001,0.000000316]")   # Universal ALP-fermion coupling
        elif i == 9:
            jobcard.set_output_launch("ALP_axW-_scan_9")
            jobcard.configure_cards()
            jobcard.add_auto_width("WALP")  # Automatically compute ALP width from decay formulas
            jobcard.add_parameter_scan("Ma", "[5.62]")      # ALP mass in GeV
            jobcard.add_parameter_scan("fa", "[1000]")     # ALP decay constant in GeV
            jobcard.add_parameter_scan("CaPhi", "[0.000316,0.0001,0.0000316,0.00001,0.00000316,0.000001,0.000000316,0.0000001]")   # Universal ALP-fermion coupling
        elif i == 10:
            jobcard.set_output_launch("ALP_axW-_scan_10")
            jobcard.configure_cards()
            jobcard.add_auto_width("WALP")  # Automatically compute ALP width from decay formulas
            jobcard.add_parameter_scan("Ma", "[10]")      # ALP mass in GeV
            jobcard.add_parameter_scan("fa", "[1000]")     # ALP decay constant in GeV
            jobcard.add_parameter_scan("CaPhi", "[0.0001,0.0000316,0.00001,0.00000316,0.000001,0.000000316,0.0000001,0.0000000316]")   # Universal ALP-fermion coupling
        elif i == 11:
            jobcard.set_output_launch("ALP_axW-_scan_11")
            jobcard.configure_cards()
            jobcard.add_auto_width("WALP")  # Automatically compute ALP width from decay formulas
            jobcard.add_parameter_scan("Ma", "[17.8]")      # ALP mass in GeV
            jobcard.add_parameter_scan("fa", "[1000]")     # ALP decay constant in GeV
            jobcard.add_parameter_scan("CaPhi", "[0.0000316,0.00001,0.00000316,0.000001,0.000000316,0.0000001,0.0000000316]")   # Universal ALP-fermion coupling
        elif i == 12:
            jobcard.set_output_launch("ALP_axW-_scan_12")
            jobcard.configure_cards()
            jobcard.add_auto_width("WALP")  # Automatically compute ALP width from decay formulas
            jobcard.add_parameter_scan("Ma", "[31.6]")      # ALP mass in GeV
            jobcard.add_parameter_scan("fa", "[1000]")     # ALP decay constant in GeV
            jobcard.add_parameter_scan("CaPhi", "[0.00001,0.00000316,0.000001,0.000000316,0.0000001,0.0000000316]")   # Universal ALP-fermion coupling
        jobcard.add_parameter_scan("CGtil", "[0.0]")   # ALP-Gluon coupling
        jobcard.add_parameter_scan("CWtil", "[0.0]")   # ALP-W coupling
        jobcard.add_parameter_scan("CBtil", "[0.0]")   # ALP-B coupling
        # Want just ALP-fermion coupling in this example
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
        
        if not dry_run:
            mg = MadgraphInterface(
                madgraph_runner=mlr,
                jobscript_str=jobscript_str,
                param_card_str=param_card,
                run_card_str=runcard_str,
                pythia_card_str=pythia_str,
                madspin_card_str=madspin_str
            )

            mg.run()
            mg.retrieve_events()
