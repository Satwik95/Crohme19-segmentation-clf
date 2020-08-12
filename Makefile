#####################################################
# Makefile to easily execute files
#####################################################
PYTHON	= python3
BIN	= ./bin
DATA    = ./data
P1	= ./project1
#####################################################

#####################################################
# PREPARE -> extracts features for segmentation
#####################################################
INKML	= $(DATA)/Train/inkml/MfrDB
INKML	+= $(DATA)/Train/inkml/MathBrush
INKML	+= $(DATA)/Train/inkml/KAIST
INKML	+= $(DATA)/Train/inkml/HAMEX
INKML	+= $(DATA)/Train/inkml/extension
INKML	+= $(DATA)/Train/inkml/expressmatch
CSV	= $(DATA)/features.csv
BASE	= n
IMG	= ./img
#####################################################

#####################################################
# SPLITTING -> splits the segmentation_features file ( 70-30 ) 
#####################################################
CSV_TRAIN	= $(DATA)/train_features.csv
CSV_TEST	= $(DATA)/test_features.csv
SPLIT	= 0.3
####################################################

#####################################################
# TRAINING -> trains a segmentation classification model
#####################################################
MODEL	= rf
DIR	= $(DATA)/model
PKL	= $(DIR)/$(MODEL).pkl
BONUS	= n
#####################################################

#####################################################
# TESTING -> performs segmentation and classification
#            of segmented symbol
#####################################################
FEATURES = $(CSV)
PKL_2=$(PKL)
PKL_1 = ./project1/model/p1_rf.pkl
RESULT	= $(DATA)/results.csv
IMG=./data/img
#####################################################

#####################################################
# LG -> convert the results to .lg file
LG	= $(DATA)/lg
#####################################################

ifeq ($(BONUS),y)
	INKML = ./data/bonus_inkml
	CSV = ./data/bonus_features.csv
	FEATURES = $(CSV)
	PKL_1 = ./project1/model/p1_rf_bonus.pkl
	PKL_2 = ./data/model/rf_bonus.pkl
	RESULT = ./data/bonus_results.csv
	IMG = ./data/img_bonus
	LG = ./data/lg/bonus
endif

ifeq ($(TRAIN),y)
	FEATURES = $(CSV_TRAIN)
	RESULT = ./data/train_results.csv
	IMG = ./data/img_train    
	LG = ./data/lg/train
endif

ifeq ($(TEST),y)
	FEATURES = $(CSV_TEST)
	RESULT = ./data/test_results.csv
	IMG = ./data/img_test
	LG = ./data/lg/test
endif

ifeq ($(BASE)$(BONUS),yy)
	INKML = ./data/bonus_inkml
	CSV = ./data/bonus_features.csv
	FEATURES = $(CSV)
	PKL = ./data/model/rf_bonus.pkl
	RESULT = ./data/bonus_results.csv
	LG = ./data/lg/bonus_baseline
endif

ifeq ($(BASE)$(TRAIN),yy)
	FEATURES = $(CSV_TRAIN)
	RESULT = ./data/train_results.csv
	LG = ./data/lg/train_baseline
endif

ifeq ($(BASE)$(TEST),yy)
	FEATURES = $(CSV_TEST)
	RESULT = ./data/test_results.csv
	LG = ./data/lg/test_baseline
endif

#######################################################################
.PHONY: all prepare split train segment lg p1_train p1_features p1_model
#######################################################################

all:	prepare segment lg

prepare:
	$(PYTHON) $(BIN)/prepare.py \
	$(INKML) \
	--csv $(CSV) 

split:
	$(PYTHON) $(BIN)/split.py \
	$(CSV) \
	$(CSV_TRAIN) \
	$(CSV_TEST) \
	--split $(SPLIT)

train:
	$(PYTHON) $(BIN)/train.py \
	$(TRAIN) \
	$(TEST) \
	$(MODEL) \
	$(PKL) \
	--bonus $(BONUS)

segment:
	$(PYTHON) $(BIN)/segmentor.py \
	$(FEATURES) \
	--pkl_2 $(PKL_2) \
	--pkl_1 $(PKL_1) \
	--out $(RESULT) \
	--baseline $(BASE) \
	--img $(IMG)    

lg:
	$(PYTHON) $(BIN)/convert_to_lg.py \
	$(RESULT) \
	$(LG)

p1_train:	p1_features p1_model

p1_features:
	$(PYTHON) $(P1)/bin/p1_feature.py \
	--dir $(P1_train) \
	--csv $(DATA)/p1_features.csv

p1_model:
	$(PYTHON) $(P1)/bin/train.py \
	--csv $(DATA)/p1_features.csv \
	--model rf \
	--bonus $(BONUS) \
	--p2_train $(CSV_TRAIN) \
	--pkl $(P1)/model/rf.pkl \
	--gt $(DATA)/iso_GT.txt










