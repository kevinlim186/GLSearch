import src.config as config
import pymysql
import paramiko
import pandas as pd
from paramiko import SSHClient
from sshtunnel import SSHTunnelForwarder
import math
import time
import os

class Performance():
	def __init__(self):
		self.intializeServer()
		time.sleep(5)
		self.intializeConnection()
		self.baseDIR = os.getcwd()
		

	def intializeServer(self):
		server= SSHTunnelForwarder(
			(config.config['sshHost'] , 22),
			ssh_username=config.config['sshUser'],
			ssh_password=config.config['sshPass'],
			remote_bind_address=(config.config['host'], 3306),
			local_bind_address=('127.0.0.1', 1122))
		server.daemon_forward_servers = True
		server.start()
	
	def intializeConnection(self):
		self.conn = pymysql.connect(host=config.config['host'], user=config.config['dbuser'],
				passwd=config.config['dbpassword'], db=config.config['database'],
				port=1122,autocommit=True, local_infile=True)
		self.cHandler = self.conn.cursor()
	

	def insertLoggerData(self, name, budgetUsed, ela_distr, ela_level, ela_meta, basic, disp, limo, nbc, pca, ic):
		sql = '''
		insert into logger (name, budgetUsed, ela_distr, ela_level, ela_meta, basic, disp, limo, nbc, pca, ic)
		values ('{}', {}, {}, {}, {}, {}, {}, {}, {}, {}, {})
		'''.format (name, budgetUsed, ela_distr, ela_level, ela_meta, basic, disp, limo, nbc, pca, ic)

		try:
			self.cHandler.execute(sql)
		except Exception as e: 
			print(e)

	def insertPathData(self,name,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,y):
		x1 = x1 if not math.isnan(x1) else 'NULL'
		x2 = x2 if not math.isnan(x2) else 'NULL'
		x3 = x3 if not math.isnan(x3) else 'NULL'
		x4 = x4 if not math.isnan(x4) else 'NULL'
		x5 = x5 if not math.isnan(x5) else 'NULL'
		x6 = x6 if not math.isnan(x6) else 'NULL'
		x7 = x7 if not math.isnan(x7) else 'NULL'
		x8 = x8 if not math.isnan(x8) else 'NULL'
		x9 = x9 if not math.isnan(x9) else 'NULL'
		x10 = x10 if not math.isnan(x10) else 'NULL'
		x11 = x11 if not math.isnan(x11) else 'NULL'
		x12 = x12 if not math.isnan(x12) else 'NULL'
		x13 = x13 if not math.isnan(x13) else 'NULL'
		x14 = x14 if not math.isnan(x14) else 'NULL'
		x15 = x15 if not math.isnan(x15) else 'NULL'
		x16 = x16 if not math.isnan(x16) else 'NULL'
		x17 = x17 if not math.isnan(x17) else 'NULL'
		x18 = x18 if not math.isnan(x18) else 'NULL'
		x19 = x19 if not math.isnan(x19) else 'NULL'
		x20 = x20 if not math.isnan(x20) else 'NULL'
		sql = '''
		insert into historicalPath (name,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20, y)
		values ('{}',{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{})
		'''.format (name,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20, y)

		try:
			self.cHandler.execute(sql)
		except Exception as e: 
			print(e)

	def importLocalFile(self, table, directory):
		sql = '''
		LOAD DATA LOCAL INFILE '{}' INTO TABLE {}
		IGNORE 1 LINES;
		'''.format(directory, table)

		try:
			self.cHandler.execute(sql)
		except Exception as e: 
			print(e)
		