prompts=(
    # === Indoor ===
    # "A living room with a sofa, a coffee table, two armchairs, a floor lamp, and a potted plant."
    # "A bedroom with a queen-size bed, two nightstands, a table lamp, a dresser, a laundry hamper, wood floor."
    # "A bathroom with a bathtub, a vanity, a soap dispenser, a toilet, a freestanding mirror, a storage cabinet, a bath mat, and a waste bin."
    # "A playroom with a toy storage box, four rocking horse, a small wooden table, and a low stool."
    "A modern cardio station with an electric treadmill, a stationary exercise bike, a rowing machine, a standing floor fan, and a tall water dispenser."
    # "A clean waiting room with a row of three linked metal chairs, a square coffee table, a floor-standing magazine rack, a tall water dispenser, a large potted palm, and a round waste bin."
    # "A laundromat section with two front-loading washing machines, a metal folding table, a laundry cart with wheels, a large plastic laundry basket, and a snack vending machine."
    # "A bookstore corner with a double-sided wooden bookshelf, a rectangular display table, a wooden checkout counter, a digital cash register on the counter, and a small metal step stool."
    # "A home office with a standing desk, a chair, a monitor, a laptop, a filing cabinet, a floor lamp, a large plant, and a trash bin." # Needs manual adjustment
    # "A kitchen with a large countertop, three bar stools, a refrigerator, a stove, a kitchen island, and a pantry cabinet." # Needs manual adjustment
    # "A restaurant dining set with a round wooden pedestal table, four upholstered dining chairs, a wooden high chair, and a two-tier serving trolley." # Not working well
    # "A formal meeting room with a long oval conference table, six ergonomic office chairs, a mobile whiteboard on wheels, and a conference speakerphone placed on the table center." # Not working well
    # "A nostalgic music corner with an upright piano, a cylindrical piano stool, a metal sheet music stand, a drum set, a mixing console." # Poor quality

    # === Structure/Symmetric Layout ===
    # "A rectangular dining table with six chairs, three on each side."
    # "A stack of four books with a small ceramic vase on top."
    # "15 coke cans stacking in a pyramid"
    # "A grid of 4 identical plants"
    # "A road safety setup with six orange traffic cones arranged in a straight line with equal spacing."
    # "A laboratory sample station with a rectangular metal tray and six transparent glass beakers arranged in a 2x3 grid."
    # "A minimalist office layout with four identical white square desks arranged in a 2x2 grid, each with a monitor on top."
    # "A round table with eight chairs evenly spaced around it and facing towards it." This case is indeed difficult.


    # === Outdoor City/Public Space ===
    # "A campsite with a large tent, a camping chair, a camping table, a lantern placed on the table, a portable cooler, and a camping stove."
    # "A beach relaxation area with a beach umbrella, two lounge chairs, a cooler, a beach blanket, two towels, a small table, and a beach bag."
    # "A community chess area with a stone chess table, two benches, a chess set, a small umbrella, two backpacks, a water jug, and a scorecard holder."
    # "A street scene with two rows of four houses, four cars, and four trees arranged along the road, cement floor texture."
    # "A minimalist city bus stop with a metal shelter frame, a long metal bench, a cylindrical trash can, a standing route sign, and a bicycle on its kickstand."
    # "A picnic scene with a picnic blanket, a picnic basket, two wine glasses, two bottles of wine, and a small table."
    
    #=== Open Vocabulary ===
    # "a Pikachu sitting on the grass, a Poké Ball standing in front of it, and an empty bench beside them."
    # "A Lego classroom scene with one Lego teacher and four Lego students, a Lego chalkboard, four Lego desks, a Lego bookshelf, two Lego books"
    # "A Japanese street corner scene with a two-story detached house, a mailbox in front of the house, a Doraemon standing in front of the house, and a utility pole beside it, cement floor texture."
    # "A medieval alchemy scene, with a heavy wooden table, a small bronze cauldron on the table, a stack of two old parchment books, a glass distilling flask, and a wooden high stool beside the table."
    # "A gardening tool set, with a wooden bench, a succulent pot on the bench, a pair of metal pruning shears, a green watering can, and a wooden crate filled with soil on the ground."
    # "A fairy-tale garden corner with a rusty vintage water pump, a wooden cart filled with red flowers, a Totoro-shaped stone statue, a wooden watering can, three clay flower pots, and a small wooden ladder."
    # "A miniature Lego theme park scene with a large brick-built dinosaur, two Lego minifigures, a brick-built tree, a brick-built fountain, a Lego park bench, and a brick-built trash can."
    # "A post-apocalyptic survival cache with a rusty fuel drum, two stacked plastic crates, a gas mask on the crate, a machete, an old radio station, and a worn-out canvas bag."
    # "A music nook with a wooden low cabinet, a vinyl record player on top, a vinyl record beside it, and two large floor-standing speakers placed symmetrically on both sides of the cabinet."
)


OUTPUT_ROOT="outputs"

# RESUME_FROM_DIR: Original directory to resume from.
# RESUME_STEP: Step to resume from. For debugging or human editing. If this variable is not enabled, the process will start from step 1 by default.
# There are 20 steps in total, with i ranging from 1 to 20. The state before step i equals to the state after step i-1. step_i.png and step_i_state.json represent the state before step i. Correspondingly, "resume from step i" means returning to the state before step i and re-executing step i.
# RESUME_FROM_DIR=$OUTPUT_ROOT/a_campsite_with_a_large_tent_a_camping_chair_a_cam_20260217_194726
# RESUME_STEP=6

# HUMAN_EDITING_STEP: Human-edit mode starts at this step, requiring feedback on each step. If this variable is not enabled, human feedback is not required by default. Currently, human editing for only a specific step is not supported, where the agent would then automatically complete the remaining tasks. A possible workaround is that after HUMAN_EDITING_STEP, the human message input for each step is left empty. If this functionality is needed, you can modify the code yourself to implement it.
# A typical application scenario is running the entire process automatically and, upon reaching max steps, realizing that adjustments are needed. In this case, you can enable human-edit mode starting from a certain step and make adjustments step by step until satisfied. You need to set RESUME_FROM_DIR, and you can set RESUME_STEP and HUMAN_EDITING_STEP to the same value.
# HUMAN_EDITING_STEP=6



for prompt in "${prompts[@]}"; do
    echo "Processing prompt: '$prompt'"

    if [ -n "$RESUME_FROM_DIR" ]; then
        original_dir_name=$(basename "$RESUME_FROM_DIR")
        timestamp=$(date +"%Y%m%d_%H%M%S")
        dest_dir="${OUTPUT_ROOT}/${timestamp}_${original_dir_name}"
        echo "Resume from: ${RESUME_FROM_DIR}"
        echo "New Output Directory: ${dest_dir}"
        mkdir -p "${dest_dir}"
        if [ -d "${RESUME_FROM_DIR}/assets" ]; then
            cp -r "${RESUME_FROM_DIR}/assets" "${dest_dir}/"
            echo "Copied assets/"
        fi
        if [ -d "${RESUME_FROM_DIR}/logs" ]; then
            cp -r "${RESUME_FROM_DIR}/logs" "${dest_dir}/"
            echo "Copied logs/"
        fi
        if [ -n "$RESUME_STEP" ]; then
            for ((i=1; i<=RESUME_STEP; i++)); do
                # Copy image files.
                step_img=$(printf "%s/step_%03d.png" "$RESUME_FROM_DIR" "$i")
                if [ -f "$step_img" ]; then
                    cp "$step_img" "${dest_dir}/"
                    echo "Copied step_$(printf '%03d' $i).png"
                fi

                step_state=$(printf "%s/step_%03d_state.json" "$RESUME_FROM_DIR" "$i")
                if [ -f "$step_state" ]; then
                    cp "$step_state" "${dest_dir}/"
                    echo "Copied step_$(printf '%03d' $i)_state.json"
                fi
            done
        fi

    else
        dest_dir="${prompt,,}"
        dest_dir="${dest_dir//[^a-z0-9]/_}"
        dest_dir=$(echo "$dest_dir" | sed 's/__*/_/g')
        dest_dir="${dest_dir/#_/}"
        dest_dir="${dest_dir/%_/}"
        dest_dir="${dest_dir:0:50}"

        timestamp=$(date +"%Y%m%d_%H%M%S")
        dest_dir="${OUTPUT_ROOT}/${dest_dir}_${timestamp}"
        echo "Output Directory: ${dest_dir}"
    fi

    python main.py \
        --model gemini \
        --output-dir "${dest_dir}" \
        --prompt "$prompt" \
        ${RESUME_STEP:+--resume-step $RESUME_STEP} \
        ${HUMAN_EDITING_STEP:+--human-editing-step $HUMAN_EDITING_STEP}
    
    source visualize.sh ${dest_dir}

done

# Clear resume-related variables so the next prompt is not affected.
RESUME_FROM_DIR=""
RESUME_STEP=""
HUMAN_EDITING_STEP=""