# -*- coding: utf-8 -*-
FFT_SIZE = 1024
SAMPLING_RATE = int(20e6)
CENTER_FREQ = 2.447e9
NUM_FFT_SPEC = 1500
RECORDING_TIME = 15

# Define the hierarchy
make_labels = {
        "Pro-OFDM": 0,
        "OFDM": 1,
        "Wi-Fi": 2
    }

type_labels = {
        "Lightbridge": 0,
        "OcuSync": 1,
        "SkyLink": 2,
        "HD video": 3,
        "Wi-Fi 4": 4,
    }

hierarchy = {
        "Pro-OFDM": {
            "Lightbridge": [
                "DJI Inspire 1",
                "DJI Inspire 2", 
                "DJI Phantom 3 Adv",
                "DJI Phantom 4",
            ],
            "OcuSync": [
                "DJI Phantom 4 Pro V2.0",
                "DJI Mavic Pro",
                "DJI Mavic 2 Pro",
                "DJI Mavic 3",
                "DJI Mavic 4",
                "DJI Mavic Mini 4",
                "DJI Mini 3",
                "DJI Air 2S",
                "DJI FPV",
            ],
            "SkyLink": [
                "Autel EVO II",
            ],
            "HD video": [       
                "Autel X-Star Premium"
            ],
        },
        "OFDM": {
            "Wi-Fi 4": [    
                "Holystone HS 720E",
                "Holystone HS 100G",
            ]
        },
        "Wi-Fi": {
            "Wi-Fi 4": [
                "DJI Mavic Air",
                "DJI Spark",
                "DJI Phantom 3 Standard",
                "DJI Mavic Mini 1",
                "Yuneec Q500-HD",
                "Parrot Anafi",
                "Ruko F11 GIM",
                "Hbson X4 Air",
                "DJI Tello",
                "Skydio 2",
            ]
        },
    }

