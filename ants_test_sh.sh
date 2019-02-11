#!/bin/sh

# path to atlases
atlas="/data/vision/polina/users/adalca/MNI_analysis_Guha/buckner61_in_MNI_ANTs_warped_norm.nii.gz"
atlas_seg="/data/vision/polina/users/adalca/MNI_analysis_Guha/buckner61_in_MNI_ANTs_warped_norm_seg.nii.gz"

# some paths and settings
input_vol_path="/data/ddmg/voxelmorph/data/t1_mix/proc/resize256-crop_x32/validate/vols/"
input_seg_path="/data/ddmg/voxelmorph/data/t1_mix/proc/resize256-crop_x32/validate/asegs/"
out_path_init="/data/vision/polina/projects/ADNI/work/neuron/output/registration/10_23_2018/"
tmp_path="${out_path_init}/tmp/"
val_data_file="${out_path_init}/validation_pairs_with_segs.txt"

# whether to use atlas, or subject to subject
use_atlas=1
if [ ${use_atlas} -eq 1 ] ; then
    val_data_file="${out_path_init}/test_brains_unique.txt"
fi

# metric for 
  # CC or MSQ or MI. see https://sourceforge.net/p/advants/svn/5/tree/Documentation/ants.pdf?format=raw
metric="CC"

# output path
out_path="${out_path_init}/${metric}_902/"

# command paths
ants_cmd="/data/vision/polina/shared_software/ANTS/build/bin/ANTS"  # registration
warp_cmd="/data/vision/polina/shared_software/ANTS/build/bin/WarpImageMultiTransform"  # warp forward
qsub_run="/data/vision/polina/users/adalca/patchSynthesis/subspace/git-papago/sge/qsub-run"  # queue system
npz2nii="/data/vision/polina/projects/ADNI/work/neuron/output/registration/code/npz2nii.py"

##############################################################################
# running code

# prepare folders
mkdir -p ${tmp_path}
mkdir -p ${out_path}

# go through filenames

while read -r line; do
    # get moving images and segs
    moving_subj_full_path=`echo $line | cut -d "," -f 1`
    moving_subj=`basename ${moving_subj_full_path}`
    moving_seg_full_path=`echo $line | cut -d "," -f 2`
    moving_seg=`basename ${moving_seg_full_path}`

    moving_file_npz=${moving_subj_full_path}  # ${input_vol_path}/
    moving_seg_file_npz=${moving_seg_full_path}  # ${input_seg_path}/
    moving_file=${tmp_path}/${moving_subj}.nii.gz
    moving_seg_file=${tmp_path}/${moving_seg}.nii.gz

    # get fixed images and segs
    if [ ${use_atlas} -eq 1 ] ; then
        fixed_file=${atlas}
        fixed_seg_file=${atlas_seg}
        fixed_subj=`basename ${atlas}`
        fixed_seg=`basename ${fixed_seg_file}`
    else
        fixed_subj=`echo $line | cut -d "," -f 2`
        fixed_seg=`echo $line | cut -d "," -f 4`
        fixed_file_npz=${input_vol_path}/${fixed_subj}
        fixed_seg_file_npz=${input_seg_path}/${fixed_seg}
        fixed_file=${tmp_path}/${fixed_subj}.nii.gz
        fixed_seg_file=${tmp_path}/${fixed_seg}.nii.gz
    fi

    # prepare the core of the output folder
    output_core="${out_path}/${moving_subj}___to___${fixed_subj}"
    
    # save files from npz to nii.gz
    if ! [ -f ${moving_file} ] ; then
        python ${npz2nii} ${moving_file_npz} ${moving_file}
    fi
    if ! [ -f ${moving_seg_file} ] ; then
        python ${npz2nii} ${moving_seg_file_npz} ${moving_seg_file}
    fi

    # if the output of ANTs exists, skip this file
    if [ -f ${output_core}/antsWarp.nii.gz ] ; then
        echo "skipping since found ${output_core}/antsWarp.nii.gz present"
        continue;
    fi

    # create sge files
    sgeopath="${output_core}/sge/"
    mkdir -p ${sgeopath}
    sge_par_o="--sge \"-o ${sgeopath}\""
    sge_par_e="--sge \"-e ${sgeopath}\""
    sge_par_l="--sge \"-l mem_free=10G \""
    cmd_core="${sgeopath}/ants_runs_${moving_subj}_${fixed_subj}"
    cmd_file="${cmd_core}_cmd_v2.sh"
    sgerunfile="${cmd_core}_sge_v2.sh"

    # prepare ANTs command

    # prepare metric
    if [ $metric = "CC" ]
    then
        metric_opt="CC[${fixed_file},${moving_file},1,4]"
    else
        metric_opt="MSQ[${fixed_file},${moving_file},1,0]"
    fi

    # prepare iterations
    affine_iters="0x0"  # no affine, since our data is already affinely registered
    nonrigid_iters="201x201x201"  # 201 iter for 3 scales

    # prepare params
    syn_param="0.25"
    gauss_param="9,0.2"

    # prepare ANTs command. This gives us the warp(s)
    ants_subj_cmd="${ants_cmd} 3 -m ${metric_opt} -t Syn[${syn_param}] -o ${output_core}/ants --number-of-affine-iterations ${affine_iters} -i ${nonrigid_iters} -r Gauss[${gauss_param}]"
    timed_ants_subj_cmd="(time $ants_subj_cmd) &> ${output_core}/antsTime.txt"

    # prepare the warp commands. This *warps* the volumes
    warp_subj_cmd="${warp_cmd} 3 ${moving_file} ${output_core}/${moving_subj}_warped.nii.gz -R ${fixed_file} ${output_core}/antsWarp.nii.gz ${output_core}/antsAffine.txt"
    warp_seg_cmd="${warp_cmd} 3 ${moving_seg_file} ${output_core}/${moving_seg}_seg_warped.nii.gz -R ${fixed_file} ${output_core}/antsWarp.nii.gz ${output_core}/antsAffine.txt --use-NN"
    cp_seg_cmd="cp ${fixed_seg_file} ${output_core}/${fixed_seg}_seg.nii.gz"

    # prepare jobs for queue 
    echo "#SOF" > ${cmd_file}
    #echo ${timed_ants_subj_cmd} >> ${cmd_file}
    #echo ";" >> ${cmd_file}
    echo ${warp_subj_cmd} >> ${cmd_file}
    echo ${warp_seg_cmd} >> ${cmd_file}
    echo ${cp_seg_cmd} >> ${cmd_file}
    chmod a+x ${cmd_file}

    # prepare sge run file
    cmd="${qsub_run} -c $sge_par_o $sge_par_e $sge_par_l ${cmd_file} > ${sgerunfile}"
    # echo $cmd
    eval $cmd

    # run training
    sgecmd="qsub ${sgerunfile}"
    chmod a+x ${sgerunfile}
    echo -e "$sgecmd\n"
    $sgecmd

    sleep 1
done < ${val_data_file}