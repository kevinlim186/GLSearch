import src.config as config
import mysql.connector
import paramiko
import pandas as pd
from paramiko import SSHClient
from sshtunnel import SSHTunnelForwarder, socket
import math
import time
import os
import pandas as pd
import time 


class Performance():
    def __init__(self):
        if config.config['allowSSH'] == 'True':
            self.intializeServer()
            time.sleep(5)

        if config.config['allowSql'] == 'True':
            self.intializeConnection()
        
        self.baseDIR = os.getcwd()
        self.ertPerformance = pd.DataFrame(columns=['name', 'fce'])
        self.elaFeatures = pd.DataFrame()
        self.selected_checkpoint = pd.DataFrame(columns=['model', 'function', 'instance','trial', 'dimension', 'budget', 'local'])

    def intializeServer(self):
        sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        server= SSHTunnelForwarder(
            (config.config['sshHost'] , 22),
            ssh_username=config.config['sshUser'],
            ssh_password=config.config['sshPass'],
            remote_bind_address=(config.config['host'], 3306),
            local_bind_address=('127.0.0.1', 1122),
            ssh_proxy=sock)
        server.daemon_forward_servers = True
        server.start()
    
    def intializeConnection(self):
        if config.config['allowSSH'] == 'True':
            self.conn = mysql.connector.connect(host='127.0.0.1', user=config.config['dbuser'],
                passwd=config.config['dbpassword'], db=config.config['database'],
                port=1122,autocommit=True, allow_local_infile=True)
        else:
            self.conn = mysql.connector.connect(host=config.config['host'], user=config.config['dbuser'],
                passwd=config.config['dbpassword'], db=config.config['database'],
                port=config.config['port'],autocommit=True, allow_local_infile=True,force_ipv6=True)
        self.cHandler = self.conn.cursor()
    

    def insertSelectedCheckpoint(self, model, function, instance, trial, dimension, budget, local):
            self.selected_checkpoint  = self.selected_checkpoint .append({'model':model, 'function':function, 'instance': instance, 'budget': budget,'local':local, 'trial':trial , 'dimension':dimension}, ignore_index=True)
    
    def insertELAData(self, name, elaFeat):
        if 'None' in elaFeat.keys():
            elaFeat.pop('None')
        if 'inf' in elaFeat.keys():
            elaFeat.pop('inf')
        
        if config.config['allowSql'] == 'True':
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

        elaFeat['name'] = name
        self.elaFeatures = self.elaFeatures.append(elaFeat, ignore_index=True)


    def importHistoricalPath(self, directory):
        if config.config['allowSql'] == 'True':
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

    def insertPerformance(self, name, ert8, ert7, ert6, ert5, ert4, ert3, ert2, ert1, ert0, ertp1, ertp2, ertp3, ertp4, fce):
        if ert8 == None or ert8 == 'inf':
            ert8 = 'NULL'
        if ert7 == None or ert7 == 'inf':
            ert7 = 'NULL'
        if ert6 == None or ert6 == 'inf':
            ert6 = 'NULL'
        if ert5 == None or ert5 == 'inf':
            ert5 = 'NULL'
        if ert4 == None or ert4 == 'inf':
            ert4 = 'NULL'
        if ert3 == None or ert3 == 'inf':
            ert3 = 'NULL'
        if ert2 == None or ert2 == 'inf':
            ert2 = 'NULL'
        if ert1 == None or ert1 == 'inf':
            ert1 = 'NULL'
        if ert0 == None or ert0 == 'inf':
            ert0 = 'NULL'
        if ertp1 == None or ertp1 == 'inf':
            ertp1 = 'NULL'
        if ertp2 == None or ertp2 == 'inf':
            ertp2 = 'NULL'
        if ertp3 == None or ertp3 == 'inf':
            ertp3 = 'NULL'
        if ertp4 == None or ertp4 == 'inf':
            ertp4 = 'NULL'
        
        if config.config['allowSql'] == 'True':
            sql = '''
            insert into performance (name, fce, ert)
            values ('{}', {}, {})
            '''.format (name, fce, ert8)

            try:
                self.cHandler.execute(sql)
            except Exception as e: 
                print(e)
        

        self.ertPerformance = self.ertPerformance.append({'name': name, 'fce':fce, 'ert-8': ert8, 'ert-7': ert7, 'ert-6': ert6, 'ert-5': ert5, 'ert-4': ert4, 'ert-3': ert3, 'ert-2': ert2, 'ert-1': ert1, 'ert0':ert0 , 'ert1': ertp1, 'ert2':ertp2, 'ert3':ertp3, 'ert4':ertp4}, ignore_index=True)

    def saveToCSVPerformance(self, fileName):
        self.checkDirectory(self.baseDIR+ '/perf')
        fileName = self.baseDIR+ '/perf/'+ str(fileName)  
        self.ertPerformance.to_csv(fileName+ '_performance.csv',index=False)
        
    def saveToCSVELA(self,fileName):	
        self.checkDirectory(self.baseDIR+ '/perf')
        fileName = self.baseDIR+ '/perf/'+ str(fileName)  
        self.elaFeatures.to_csv(fileName+ '_elaFeatures.csv',index=False)

    def checkDirectory(self, path):
        if not os.path.exists(str(path)):
            os.mkdir(path)

    def saveToSelectedCheckPoint(self,fileName):	
        self.checkDirectory(self.baseDIR+ '/perf')
        fileName = self.baseDIR+ '/perf/selected_checkpoints'+ str(fileName)  
        self.selected_checkpoint.to_csv(fileName+ 'd.csv',index=False)
