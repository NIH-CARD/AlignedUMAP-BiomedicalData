from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfform
from reportlab.lib.utils import ImageReader
from PIL import Image
from reportlab.lib.colors import magenta, pink, blue, green, black, white
from reportlab.platypus import Paragraph, Frame

text_details = {
    'R6': ["Parkinson's disease progression ", "(PPMI clinical assessment)", "Sample count = 200", "Subtype"],
    'R5': ["Alzheimer's disease progression ", "(ADNI clinical assessment)", "Sample count = 241", "Subtype"],
    'R4': ["Bleomycin lung disease and regeneration ", "(whole lung scRNA)", "Cell count = 2,738", "Cell type"],
    'R3': ["Parkinson's and Alzheimer's disease progression", "(PPMI + ADNI brain T1 MRI)", "Sample count = 884", "Gender+Diagnosis"],
    'R2': ["MIMIC-III critical care database", "(ICU unit outcome)", "Sample count = 3,543", "Last care ICU"],
    'R1': ["COVID-19 disease severity ", "(plasma proteins PEA)", "Sample count = 383", "Disease severity"],
    'R0': ["iPSC-derived neurons  ", "(proteomics)", "Cell count = 18", "Bioreactor culture type"],
}
def add_to_c(c, index, img1, img2, img3, img4, img5, ratio=0):
    start_height = 190 * index
    # 2 - 188
    start_height += 2
    width = 195
    height = 150
    legend_height = 30
    text_height = 18
    text_list = text_details[f"R{index}"]
    if index == 1:
        icon = ImageReader('covid19.png')
    elif index == 2:
        icon = ImageReader('icu.png')
    elif index == 3:
        icon = ImageReader(f"brain.png")
    elif index == 0:
        icon = ImageReader(f"ipsc.png")
    elif index == 4:
        icon = ImageReader(f"mouse.png")
    else:
        icon = ImageReader(f"human.png")
    c.drawImage(icon, 5, start_height+height+18, width=18, height=15, preserveAspectRatio=False, showBoundary=False, mask='auto')

    c.drawImage(img2, 5, start_height, width=width, height=height, preserveAspectRatio=True, showBoundary=False, mask='auto')
    c.drawImage(img3, width+5, start_height, width=width, height=height, preserveAspectRatio=True, showBoundary=False, mask='auto')
    c.drawImage(img4, width * 2+5, start_height, width=width, height=height, preserveAspectRatio=True, showBoundary=False, mask='auto')

    c.drawImage(img1, 5, start_height + height, width=140, height=18, preserveAspectRatio=True, showBoundary=False, mask='auto')
    leg_width = int(ratio*15)
    c.drawImage(img5, 3*width - leg_width - 2, start_height +  height, width=leg_width, height=15, preserveAspectRatio=True, showBoundary=False, mask='auto')
    # c.drawImage(img5, 1.6*width, start_height +  height, width=leg_width, height=15, preserveAspectRatio=True, showBoundary=False, mask='auto')

    form = c.acroForm
    # 160 - 38
    form.textfield(name=f'fname1_{index}', value=text_list[0], tooltip='First Name', x=25, y=start_height + height+18, height=15, fontSize=9, width=4.7 * len(text_list[0]), borderColor=white, borderWidth=0, fillColor=white, textColor=black, forceBorder=False, fontName='Helvetica-Bold')
    form.textfield(name=f'fname3_{index}', value=text_list[1], tooltip='Samples Name', x=4.7 * len(text_list[0])+20, y=start_height + height+17, height=15, fontSize=8, width=4.25 * len(text_list[1]), borderWidth=0, borderColor=white, fillColor=white, textColor=black, forceBorder=False)# , fontName='Times-Bold')

    form.textfield(name=f'fname2_{index}', value=text_list[2], tooltip='Samples Name', x=160, y= start_height + height, height=15, fontSize=8, width=120, borderColor=white, borderWidth=0, fillColor=white, textColor=black, forceBorder=False)# , fontName='Times-Bold')
    form.textfield(name=f'fname4_{index}', value=text_list[3], tooltip='Samples Name', x=3*width - leg_width - 2, y=start_height + height+15, height=15, fontSize=8, width=leg_width, borderColor=white, fillColor=white, textColor=black, forceBorder=False)# , fontName='Times-Bold')
    # form.textfield(name=f'fname4_{index}', value=text_list[3], tooltip='Samples Name', x=width*2, y=start_height + height+15, height=15, fontSize=8, width=leg_width, borderColor=white, fillColor=white, textColor=black, forceBorder=False)# , fontName='Times-Bold')

    # form.textfield(name=f'fname1_{index}', value=text_list[0], tooltip='First Name', x=25, y=start_height + height + 18,
    #                height=18, fontSize=8, width=160, borderColor=white, fillColor=white, textColor=black,
    #                forceBorder=False, fontName='Helvetica-Bold')
    # form.textfield(name=f'fname3_{index}', value=text_list[1], tooltip='Samples Name', x=185,
    #                y=start_height + height + 18, height=18, fontSize=8, width=110, borderColor=white, fillColor=white,
    #                textColor=black, forceBorder=False)  # , fontName='Times-Bold')
    # form.textfield(name=f'fname2_{index}', value=text_list[2], tooltip='Samples Name', x=160, y=start_height + height,
    #                height=18, fontSize=8, width=120, borderColor=white, fillColor=white, textColor=black,
    #                forceBorder=False)  # , fontName='Times-Bold')
    # form.textfield(name=f'fname4_{index}', value=text_list[3], tooltip='Samples Name', x=width * 1.6,
    #                y=start_height + height + 15, height=15, fontSize=8, width=leg_width, borderColor=white,
    #                fillColor=white, textColor=black, forceBorder=False)  # , fontName='Times-Bold')


    c.drawBoundary(sb=1, y=190 * index + 1, x=3, w=590, h=188)


img = Image.open('python-logo.png')
# c = canvas.Canvas('mainimage.pdf', pagesize=(595.27, 841.69))
c = canvas.Canvas('mainimage.pdf', pagesize=(595.27, 1341.69))
l = ['R0', 'R1', 'R2', 'R5', 'R6', 'R3', 'R4']
l = ['R5', 'R6', 'R4', 'R3', 'R2', 'R1', 'R0']
for index, row in enumerate(l):
    img1 = ImageReader(f"allimages/{row}/img1.png")
    img2 = ImageReader(f"allimages/{row}/img2.png")
    img3 = ImageReader(f"allimages/{row}/img3.png")
    img4 = ImageReader(f"allimages/{row}/img4.png")
    img5 = ImageReader(f"allimages/{row}/img5.png")
    x = Image.open(f"allimages/{row}/img5.png")
    print (x.size[0]/float(x.size[1]))
    add_to_c(c, index, img1, img2, img3, img4, img5, ratio = x.size[0]/float(x.size[1]))
c.save()