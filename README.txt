ʵ�黷����
Python 2.x �� Python 3.x

��Ҫ�⣺
matplotlib
numpy
math
re

��Ҫ�ļ���
dataset.txt	���ݼ�����������ź�������ɣ����磺B1 A Course on Integral Equations

stop_words.txt	ͣ�ôʣ�����ȥ���������������޹ص�ͣ�ôʣ����磺"a", "on", "of"...
new_data.txt	�����ݣ���������ź�������ɣ����ڲ���LSI�ĸ���
README.txt	˵���ĵ�
lsi_test.py	LSIʵ�ִ���

��Ҫ�ļ���
A.txt		���ڴ�� �ؼ���-���� ����A��ÿһ��Ϊһ���ؼ����ڸ��������г��ֵĴ�Ƶ��
A_k.txt		���ڴ��A�� k-�Ƚ��ƾ��� ��ÿһ��Ϊһ��������
new_A.txt	���ڴ�Ÿ���������� �ؼ���-���� ����A
new_A_k.txt	���ڴ�Ÿ���������A�� k-�Ƚ��ƾ���

ʹ�÷�����
�������£�	1.�л���LSI_test�ļ�����
		2.ִ�� python ./lsi_test.py
����ʹ��Python IDE����

�޸Ĳ�����
��lsi_test.py�ļ���ײ��ҵ�
if __name__ == '__main__':
    lsi(book_names_file="./dataset.txt",     # ==>���ݼ��ļ�·��
        stop_words_file="./stop_words.txt",  # ==>ͣ�ô��ļ�·��
        save_A="./A.txt",                    # ==>�ؼ���-���� ����A����·��
        save_A_k="./A_k.txt",                # ==>A�� k-�Ƚ��ƾ��󱣴�·��
        save_new_A="./new_A.txt",            # ==>����������� �ؼ���-���� ����A����·��
        save_new_A_k="./new_A_k.txt",        # ==>����������A�� k-�Ƚ��ƾ��󱣴�·��
        theta=0.4,                           # ==>k-�Ƚ��ƾ�����ֵ��Ӱ��k��ȡֵ�����ڿ��ӻ�ʱֻ����ʵ2ά���ݣ����k=2ʱ���ӻ������Ϊ׼ȷ�������޸���ʱ����theta=0.4���и���ʱ����theta=0.3
        cos_treshold=0.9,                    # ==>�н����Ҿ�����ֵ��ֵԽ�����ƶ�Խ��
        # query='Application and theory',
        query=None,                          # ==>��ѯ���ԣ���ѯ��������ص��������������������cos_treshold���ƣ�������None�ر�
        # update_file="./new_data.txt")
        update_file=None)                    # ==>������·�������ڸ��²��ԣ�������None�ر�

��ע������ֻ�ܿ��ӻ�2ά���ݣ���thetaȡֵ������k>2ʱ���ܻ���ֿ��ӻ���������ļн����ҽ������������k<2ʱ�޷���ά���ӻ���������ʹ��k=2