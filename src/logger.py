import src.config as config
import pymysql
import paramiko
import pandas as pd
from paramiko import SSHClient
from sshtunnel import SSHTunnelForwarder
import math
import time
import os
import pandas as pd
import time 

class Performance():
	def __init__(self):
		if config.config['instance'] == 'local':
			self.intializeServer()
			time.sleep(5)

		if config.config['instance'] != 'remote':
			self.intializeConnection()
		
		self.baseDIR = os.getcwd()
		self.ertPerformance = pd.DataFrame(columns=['name', 'fce', 'ert'])
		self.elaFeatures = pd.DataFrame()

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
		if config.config['instance'] == 'local':
			self.conn = pymysql.connect(host=config.config['host'], user=config.config['dbuser'],
				passwd=config.config['dbpassword'], db=config.config['database'],
				port=1122,autocommit=True, local_infile=True)
		else:
			self.conn = pymysql.connect(host=config.config['host'], user=config.config['dbuser'],
				passwd=config.config['dbpassword'], db=config.config['database'],
				port=config.config['port'],autocommit=True, local_infile=True)
		self.cHandler = self.conn.cursor()
	

	def insertELAData(self, name, elaFeat):
		if 'None' in elaFeat.keys():
			elaFeat.pop('None')
		if 'inf' in elaFeat.keys():
			elaFeat.pop('inf')
		
		if config.config['instance'] != 'remote':
			columns = ', '.join('`'+ e + '`' for e in elaFeat.keys())
			values = ', '.join(str(e) for e in elaFeat.values()).replace('nan', 'NULL')
			sql = '''
			insert into elaFeatures (name, {})
			values ('{}', {})
			'''.format (columns, name, values)

			try:
				self.cHandler.execute(sql)
			except Exception as e: 
				print(e)
		else:
			self.elaFeatures = self.elaFeatures.append(elaFeat, ignore_index=True)


	def importHistoricalPath(self, directory):
		if config.config['instance'] != 'remote':
			sql = '''
			LOAD DATA LOCAL INFILE '{}' INTO TABLE historicalPath
			FIELDS TERMINATED BY ','
			IGNORE 1 LINES
			(@x1,@x2,@x3,@x4,@x5,@x6,@x7,@x8,@x9,@x10,@x11,@x12,@x13,@x14,@x15,@x16,@x17,@x18,@x19,@x20,y,name)
			set
			x1 = NULLIF(@x1,''),
			x2 = NULLIF(@x2,''),
			x3 = NULLIF(@x3,''),
			x4 = NULLIF(@x4,''),
			x5 = NULLIF(@x5,''),
			x6 = NULLIF(@x6,''),
			x7 = NULLIF(@x7,''),
			x8 = NULLIF(@x8,''),
			x9 = NULLIF(@x9,''),
			x10 = NULLIF(@x10,''),
			x11 = NULLIF(@x11,''),
			x12 = NULLIF(@x12,''),
			x13 = NULLIF(@x13,''),
			x14 = NULLIF(@x14,''),
			x15 = NULLIF(@x15,''),
			x16 = NULLIF(@x16,''),
			x17 = NULLIF(@x17,''),
			x18 = NULLIF(@x18,''),
			x19 = NULLIF(@x19,''),
			x20 = NULLIF(@x20,'')
			;
			'''.format(directory)

			try:
				self.cHandler.execute(sql)
			except Exception as e: 
				print(e)
		
	def insertPerformance(self, name, ert, fce):
		if ert == None or ert == 'inf':
			ert = 'NULL'
		
		if config.config['instance'] != 'remote':
			sql = '''
			insert into performance (name, fce, ert)
			values ('{}', {}, {})
			'''.format (name, fce, ert)

			try:
				self.cHandler.execute(sql)
			except Exception as e: 
				print(e)
		
		else:
			self.ertPerformance = self.ertPerformance.append({'name': name, 'fce':fce, 'ert': ert}, ignore_index=True)

	def saveToCSV(self):
		fileName = self.baseDIR+ '/temp/'+ str(round(time.time()))  
		self.ertPerformance.to_csv(fileName+ '_performance.csv',index=False)
		self.elaFeatures.to_csv(fileName+ '_elaFeatures.csv',index=False)