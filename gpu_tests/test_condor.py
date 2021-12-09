#!/home/miranda9/miniconda3/envs/automl-meta-learning/bin/python

"""
Locations of python for each cluster

vision
#!/home/miranda9/miniconda3/envs/automl-meta-learning/bin/python

HAL
#!/home/miranda9/.conda/envs/automl/bin/python3.7

todo later, how to have both clusters have conda save to the same path, for now fix it with a symbolic link

not work:
ln -s /home/miranda9/miniconda3/ /home/miranda9/.conda

correct:
ln -s /home/miranda9/miniconda3/envs/automl-meta-learning/bin/python /home/miranda9/.conda/envs/automl/bin/python3.7

https://askubuntu.com/a/1234530/230288
"""

print('TESTING CONDOR\a')

# def get_cluster_jobids_old(args):
#     import os
#
#     args.jobid = -1
#     args.slurm_jobid, args.slurm_array_task_id = -1, -1
#     if 'SLURM_JOBID' in os.environ:
#         args.slurm_jobid = int(os.environ['SLURM_JOBID'])
#         args.jobid = args.slurm_jobid
#     if 'SLURM_ARRAY_TASK_ID' in os.environ:
#         args.slurm_array_task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
#     args.condor_jobid = -1
#     if 'MY_CONDOR_JOB_ID' in os.environ:
#         args.condor_jobid = int(os.environ['MY_CONDOR_JOB_ID'])
#         args.jobid = args.condor_jobid
#     return args
#
# def test_my_email():
#     from types import SimpleNamespace
#     from pathlib import Path
#     from uutils.emailing import send_email, send_email_pdf_figs
#     import os
#
#     args = SimpleNamespace()
#     ##
#     args.mail_user = 'brando.science@gmail.com'
#     args.pw_path = Path('~/pw_app.config.json').expanduser()
#     ## get cluster job ids
#     args = get_cluster_jobids_old(args)
#     ## compose message
#     args.message = f"test conder \n\nos.environ = {os.environ} \n\nargs: {args}"
#     ## send e-mail
#     send_email(subject=f'test condor my email: {args.jobid}', message=args.message, destination=args.mail_user, password_path=args.pw_path)

def test():
    import sys
    import os

    for p in sys.path:
        print(p)

    print(os.environ)


if __name__ == '__main__':
    print('running __main__')
    # test_my_email()
    test()
    print('Done!\a\n')

