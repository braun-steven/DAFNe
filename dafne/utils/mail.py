#!/usr/bin/env python3
from detectron2.utils import comm
import os
import smtplib
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from tabulate import tabulate

import torch


def notify_mail(credentials_path, subject, message, filename=None):
    """
    Sends an E-Mail to the a specified address with a chosen message and subject
     Args:
        cred_path (string): path to the credentials
        subject (string): subject of the e-mail
        message (string): message of the e-mail
        filename (string): path to the file that is going to be send via mail
    Returns:
        None
    """
    with open(credentials_path) as f:
        lines = f.readlines()
        email = lines[0].split("=")[1][:-1]
        password = lines[1].split("=")[1][:-1]
        receiver = lines[2].split("=")[1][:-1]

    msg = MIMEMultipart()

    msg["From"] = email
    msg["To"] = receiver
    msg["Subject"] = subject

    body = f"<body>{message}</body>"
    msg.attach(MIMEText(body, "html"))

    if filename is not None:
        attachment = open(filename, "rb")

        part = MIMEBase("application", "octet-stream")
        part.set_payload((attachment).read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", "attachment; filename= %s" % filename)

        msg.attach(part)

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(email, password)
    text = msg.as_string()
    server.sendmail(email, receiver, text)
    server.quit()


def send_mail_error(cfg, args, errormsg):
    if not comm.is_main_process():
        return
    subject = f"[DAFNE -- {cfg.OUTPUT_DIR}] Experiment Error!"
    message = (
        f"The experiment in {cfg.OUTPUT_DIR} has failed. An error occurred "
        f"during training:\n\n{errormsg}"
    )
    notify_mail(
        credentials_path=cfg.EMAIL.CREDENTIALS_PATH,
        subject=subject,
        message=message,
        filename=os.path.join(cfg.OUTPUT_DIR, "log.txt"),
    )


def send_mail_success(cfg, results):
    if not comm.is_main_process():
        return
    exp_name = cfg.EXPERIMENT_NAME
    subject = f"[{exp_name}] Experiment Finished!"
    message = f"Experiment: {exp_name}<br>"
    message += "Test Results:<br>"
    message += "<br>"

    # Copy to prevent modification of the original
    results = results.copy()

    for dataset_name, dataset_result in results.items():
        # Copy to prevent modification of the original
        dataset_result = dataset_result.copy()
        message += f"{dataset_name}<br>"
        message += "-" * (26) + "<br>"


        # Add mAP to the top
        table = []

        if "task1" not in dataset_result:
            continue

        mAP = dataset_result["task1"].pop("map")
        table.append(("map", mAP))
        table += list(dataset_result["task1"].items())
        message += tabulate(
            table, headers=["class", "ap"], tablefmt="html", floatfmt=("", "1.3f")
        )

        message += "<br><br>"

    notify_mail(
        subject=subject,
        message=message,
        filename=os.path.join(cfg.OUTPUT_DIR, "log.txt"),
        credentials_path=os.environ["EMAIL_CREDENTIALS"],
    )
