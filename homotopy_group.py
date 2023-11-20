from re import A, template
from flask import Flask, render_template, request #追加
from jinja2 import Template
#from openpyxl import load_workbook
import sqlite3
import numpy as np
import pandas as pd
import sympy as sp

app = Flask(__name__)

n = 2
k = 2
HoGroup = []
Arrow = []
table_group = [[], [], [], [], []]
table_gen = [[], [], [], [], []]
table_arrow = [[], [], [], [], []]
table_image = [[], [], [], [], []]
table_ref = [[], [], [], [], []]
m_d_sum = 1

@app.route('/')
def homotopy_group():
  return render_template('homotopy_group.html', title='flask test', n=n, k=k, \
    HoGroup=HoGroup, Arrow=Arrow, \
    table_group=table_group, table_gen=table_gen, table_arrow=table_arrow, \
    table_image=table_image, table_ref=table_ref, m_d_sum=m_d_sum)

@app.route('/register', methods=['post'])
def register():
  n = request.form['n']
  k = request.form['k']
  n = int(n)
  k = int(k)

  db_name = 'sphere.db'
  conn = sqlite3.connect(db_name)
  df = pd.read_csv('.\sphere.csv')
  df.to_sql('sphere', conn, if_exists='replace')
  df = pd.read_csv('.\HyoujiGen.csv')
  df.to_sql('gen', conn, if_exists='replace')
  c = conn.cursor()

  inf = float('inf')

  # dict_factoryの定義
  def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
      d[col[0]] = row[idx]
    return d

  # row_factoryの変更(dict_factoryに変更)
  c.row_factory = dict_factory

  def Dot(A, B):
    return sp.expand((A.transpose()*B)[0,0])

  class HomotopyGroup:
    def __init__(self, n, k):
      self.n = n
      self.k = k

      if k <= -1:
        query = f'select * from sphere where n = {0} and k = {-1}'
        query_count = f'select count( * ) from sphere where n = {0} and k = {-1}'
      elif k+2 >= n:
        query = f'select * from sphere where n = {n} and k = {k}'
        query_count = f'select count( * ) from sphere where n = {n} and k = {k}'
      else:
        query = f'select * from sphere where n = {k+2} and k = {k}'
        query_count = f'select count( * ) from sphere where n = {k+2} and k = {k}'
      self.query = query
      self.query_count = query_count

    def direct_sum(self): # 直和成分の個数を与えるメソッド
      for row in c.execute(self.query_count):
        return_direct_sum = row['count( * )']
      # print(f'direct_sum = {return_direct_sum}')
      return return_direct_sum

    def order_list(self): # order のリストを与えるメソッド
      try:
        ord = [row['orders'] if row['orders'] == inf else int(row['orders']) for row in c.execute(self.query)]
      except:
        ord = [0]
      return ord

    def max_direct_sum(self):
      nn = [self.n*2-1, self.n-1, self.n, self.n*2-1, self.n-1]
      kk = [self.k-self.n+2, self.k, self.k, self.k-self.n+1, self.k-1]
      m_d_sum = [HomotopyGroup(nn[i], kk[i]).direct_sum() for i in range(5)]
      return max(m_d_sum)

    def query_id(self, id):
      if self.k <= -1:
        queryid = f'select * from sphere where n = {0} and k = {-1} and id = {id}'
      elif self.k + 2 >= self.n:
        queryid = f'select * from sphere where n = {self.n} and k = {self.k} and id = {id}'
      else:
        queryid = f'select * from sphere where n = {self.k + 2} and k = {self.k} and id = {id}'
      return queryid     

    def rep_list(self, id): # 表示元を合成で表示する際の数字のリストを与えるメソッド
      c.execute(self.query_id(id))
      int_list = list(map(int, c.fetchone()['Element'].split()))
      return int_list

    def gen_coe_list(self, id): # 生成元の表示元による一次結合の係数のリストを与えるメソッド
      try:
        for row in c.execute(self.query_id(id)):
          int_list = list(map(int, row['gen_coe'].split()))
        del int_list[self.direct_sum():] # リストの長さを direct_sum の数にする
      except:
        int_list = []
      return int_list

    def el_dim_list(self, el_list): # 表示元の境目の次元のリストを与えるメソッド
      el_d = [self.n]
      for (i, elem) in enumerate(el_list):
        gen_query = f'select * from gen where id = {elem}'
        for row in c.execute(gen_query):
          el_d.append(el_d[i] + row['k'])
      return el_d

    def pi_tex(self):
      return_pi_tex = f'\pi_{ {self.n + self.k} }^{ {self.n} }'
      return return_pi_tex

    def group_structure(self):
      group = []
      for row in c.execute(self.query):
        if row['orders'] == 0:
          group.append('0')
        elif row['orders'] == inf:
          group.append('Z')
        else:
          try:
            group.append(f"Z_{ {int(row['orders'])} }")
          except:
            group.append('0')
      self_direct_sum = self.direct_sum()
      group_gen = [self.rep_linear_tex(self.gen_coe_list(j)) for j in range(self_direct_sum)]
      group_tex_list = [gr + '\{' + gen + '\}' for (gr, gen) in zip(group, group_gen)]
      return_group_tex = ' \oplus '.join(group_tex_list)
      return return_group_tex

    def el_tex(self, el_list): # 表示元の tex 表示を与えるメソッド
      el_dim_list = self.el_dim_list(el_list)
      tex = ''
      sq_count = 1
      for (i, elem) in enumerate(el_list):
        gen_query = f'select * from gen where id = {elem}'
        for row in c.execute(gen_query):
          if row['kinds'] == 0:
            if el_dim_list[i] < row['n']: # row[5] = n 
              tex = tex + '{' + row['latex'] + f'_{ {el_dim_list[i]} }' + '(Not Defined)}'
            elif i==0 or el_list[i-1] != el_list[i]: # 一番最初か、前の id と違うとき、カウントを１にする。
              sq_count = 1
            else: # 前の id と同じときにカウントを＋１
              sq_count = sq_count + 1 
            if i == len(el_list)-1: # 最後のときに tex を追加
              if el_dim_list[i] >= row['n']:
                if '{3,' in row['latex']:
                  tex = tex + '{' + row['latex'] + f'{el_dim_list[i-sq_count+1]}' + '} }'
                else:
                  tex = tex + '{' + row['latex'] + f'_{ {el_dim_list[i-sq_count+1]} }' + '}'
                if sq_count >=2:
                  tex = tex.removesuffix('}') + f'^{ {sq_count} }' + '}'
            elif el_list[i] != el_list[i+1]: # 次の id と違うときに tex を追加
              if el_dim_list[i] >= row['n']:
                if '{3,' in row['latex']:
                  tex = tex + '{' + row['latex'] + f'{el_dim_list[i-sq_count+1]}' + '} }'
                else:
                  tex = tex + '{' + row['latex'] + f'_{ {el_dim_list[i-sq_count+1]} }' + '}'
                if sq_count >=2: # カウントの数をべきにする。
                  tex = tex.removesuffix('}') + f'^{ {sq_count} }' + '}'
            # 次の id と同じ場合は何もしない
          elif el_dim_list[i] < row['n']:
            tex = tex + '{' + 'E' + f"^{ {el_dim_list[i] - row['n']} }" + row['latex'] + '(Not Defined)}'
          elif el_dim_list[i] == row['n']:
            tex = tex + '{' + row['latex'] + '}'
          elif el_dim_list[i] == row['n'] + 1:
            tex = tex + '{' + 'E ' + row['latex'] + '}'
          else:
            tex = tex + '{' + 'E' + f"^{ {el_dim_list[i] - row['n']} }" + row['latex'] + '}'

      if 0 in el_list:
        tex = '0'

      return tex

    def rep_linear_tex(self, coe, totalcoe=1):
      rep_coe = [c * totalcoe for c in coe]
      order_list = self.order_list()
      direct_sum = self.direct_sum()
      if order_list == [0]:
        rep_lin = '0'
      elif rep_coe != []:
        symbol_list = ['x00', 'x01', 'x02', 'x03', 'x04', 'x05', 'x06', 'x07', 'x08', 'x09', 'x10', 'x11']
        X = sp.Matrix([sp.Symbol(symbol_list[i]) for i in range(direct_sum)])
        A = sp.Matrix([rep_coe[i] for i in range(direct_sum)])
        rep_lin = str(Dot(X, A)).replace('*x', 'x')
        for i in range(direct_sum):
          rep_lin = rep_lin.replace(symbol_list[i], self.el_tex(self.rep_list(i)))
      else:
        rep_lin = ''
      return rep_lin

    # def rep_linear_tex(self, coe, totalcoe=1):
    #   # print(f'coe = {coe}')
    #   # rep_coe = list(map(lambda x: x*totalcoe, coe))
    #   rep_coe = [c * totalcoe for c in coe]
    #   # print('rep_linear_tex', coe, rep_coe, totalcoe)
    #   order_list = self.order_list()
    #   direct_sum = self.direct_sum()
    #   # print(f'rep_coe = {rep_coe}')
    #   if order_list == [0]:
    #     rep_lin = '0'
    #   elif rep_coe != []:
    #     symbol_list = ['x00', 'x01', 'x02', 'x03', 'x04', 'x05', 'x06', 'x07', 'x08', 'x09', 'x10', 'x11']
    #     X = sp.Matrix([sp.Symbol(symbol_list[i]) for i in range(direct_sum)])
    #     # X = sp.Matrix([sp.Symbol(self.el_tex(self.rep_list(i))) for i in range(direct_sum)])
    #     # print(f'{coe}, rep_coe = {rep_coe}')

    #     # def mod_coe(i):
    #     #   # print(f'i = {i}')
    #     #   try:
    #     #     if order_list[i] == inf:
    #     #       return rep_coe[i]
    #     #     elif rep_coe[i] % order_list[i] > order_list[i] /2:
    #     #       return rep_coe[i] % order_list[i] - order_list[i]
    #     #     else:
    #     #       return rep_coe[i] % order_list[i]
    #     #   except:# IndexError:
    #     #     return rep_coe[i]# i

    #     # print(f'{self.n}, {self.k}, direct_sum = {direct_sum}')
    #     A = sp.Matrix([rep_coe[i] for i in range(direct_sum)])
    #     # A = sp.Matrix([mod_coe(i) for i in range(direct_sum)])
    #     # print(A)
    #     rep_lin = str(Dot(X, A)).replace('*x', 'x')
    #     for i in range(direct_sum):
    #       rep_lin = rep_lin.replace(symbol_list[i], self.el_tex(self.rep_list(i)))
    #   else:
    #     rep_lin = ''
    #   return rep_lin

    def rep_to_gen_matrix(self):
      matrix_list = []
      direct_sum = self.direct_sum()
      try:
        for id in range(direct_sum):
          for row in c.execute(self.query_id(id)):
            int_list = list(map(int, row['gen_coe'].split()))
          del int_list[direct_sum:]
          matrix_list.append(int_list)
      except:
        pass
      return_matrix = sp.Matrix(matrix_list)
      return return_matrix

    # def gen_linear_tex(self, coe, totalcoe=1):
    #   # print(f'coe = {coe}')
    #   # rep_coe = list(map(lambda x: x*totalcoe, coe))
    #   rep_coe = [c * totalcoe for c in coe]
    #   # print('rep_linear_tex', coe, rep_coe, totalcoe)
    #   order_list = self.order_list()
    #   direct_sum = self.direct_sum()
    #   # print(f'rep_coe = {rep_coe}')
    #   if order_list == [0]:
    #     rep_lin = '0'
    #   elif rep_coe != []:
    #     symbol_list = ['x00', 'x01', 'x02', 'x03', 'x04', 'x05', 'x06', 'x07', 'x08', 'x09', 'x10', 'x11']
    #     X = sp.Matrix([sp.Symbol(symbol_list[i]) for i in range(direct_sum)])
    #     # X = sp.Matrix([sp.Symbol(self.el_tex(self.rep_list(i))) for i in range(direct_sum)])
    #     # print(f'{coe}, rep_coe = {rep_coe}')
    #     def mod_coe(i):
    #       # print(f'i = {i}')
    #       try:
    #         if order_list[i] == inf:
    #           return rep_coe[i]
    #         elif rep_coe[i] % order_list[i] > order_list[i] /2:
    #           return rep_coe[i] % order_list[i] - order_list[i]
    #         else:
    #           return rep_coe[i] % order_list[i]
    #       except:# IndexError:
    #         return rep_coe[i]# i
    #     # print(f'{self.n}, {self.k}, direct_sum = {direct_sum}')
    #     A = sp.Matrix([mod_coe(i) for i in range(direct_sum)])
    #     B = self.rep_to_gen_matrix()
    #     C = A * B
    #     print(f'A = {A, B, C}')
    #     rep_lin = str(Dot(X, C)).replace('*x', 'x')
    #     for i in range(direct_sum):
    #       rep_lin = rep_lin.replace(symbol_list[i], self.el_tex(self.rep_list(i)))
    #   else:
    #     rep_lin = ''
    #   return rep_lin

    def P_image_tex(self, id):
      if self.n % 2 == 1:
        P_hg = HomotopyGroup(int((self.n - 1) / 2), int((self.n + 2 * self.k - 3) / 2))
        return_P_image_tex = P_hg.rep_linear_tex(self.P_coe(id)[0])
      else:
        return_P_image_tex = ''
      return return_P_image_tex

    def E_image_tex(self, id):
      E_hg = HomotopyGroup(self.n+1, self.k)
      return_E_image_tex = E_hg.rep_linear_tex(self.E_coe(id)[0])
      return return_E_image_tex

    def H_image_tex(self, id):
      H_hg = HomotopyGroup(2*self.n-1,self.k-self.n+1)
      return_H_image_tex = H_hg.rep_linear_tex(self.H_coe(id)[0])
      return return_H_image_tex

    def rep_coe_to_id_list(self, repcoelist):
      return_id_list = [i for i in range(self.direct_sum()) if repcoelist[i] != 0]
      return_coe_list = [repcoelist[i] for i in return_id_list]
      return return_id_list, return_coe_list

    def rep_coe_to_el_list(self, repcoelist):
      id_list = self.rep_coe_to_id_list(repcoelist)
      return_el_list = [self.rep_list(i) for i in id_list[0]]
      return_coe_list = id_list[1]
      return return_el_list, return_coe_list

    def P_coe_matrix(self):
      matrix_list = []
      direct_sum = self.direct_sum()
      hg_P = HomotopyGroup((self.n - 1)/2, self.n + self.k - 2 - (self.n - 1)/2)
      P_direct_sum = hg_P.direct_sum()
      for id in range(direct_sum):
        for row in c.execute(self.query_id(id)):
          int_list = list(map(int, row['P_coe'].split()))
        del int_list[P_direct_sum:]
        matrix_list.append(int_list)
      return_P_coe_matrix = sp.Matrix(matrix_list)
      return return_P_coe_matrix

    def E_coe_matrix(self):
      matrix_list = []
      direct_sum = self.direct_sum()
      hg_E = HomotopyGroup(self.n + 1, self.k)
      E_direct_sum = hg_E.direct_sum()
      for id in range(direct_sum):
        for row in c.execute(self.query_id(id)):
          try:
            int_list = list(map(int, row['E_coe'].split()))
          except:
            int_list = [0] * 12
        del int_list[E_direct_sum:]
        if int_list != []:
          matrix_list.append(int_list)
        else:
          matrix_list.append([0] * E_direct_sum)
      return_E_coe_matrix = sp.Matrix(matrix_list)
      return return_E_coe_matrix

    def H_coe_matrix(self):
      matrix_list = []
      direct_sum = self.direct_sum()
      hg_H = HomotopyGroup((self.n * 2) - 1, self.n + self.k - (self.n * 2) + 1)
      H_direct_sum = hg_H.direct_sum()
      for id in range(direct_sum):
        for row in c.execute(self.query_id(id)):
          try:
            int_list = list(map(int, row['H_coe'].split()))
          except:
            int_list = [0] * H_direct_sum
        del int_list[H_direct_sum:]
        matrix_list.append(int_list)
      return_H_coe_matrix = sp.Matrix(matrix_list)
      return return_H_coe_matrix

    def rep_coe_to_gen_coe(self, repcoelist):
      repcoematrix = sp.Matrix([repcoelist])
      if repcoematrix == sp.Matrix([[0]]):
        direct_sum = self.direct_sum()
        return_gen_coe_list = [0] * direct_sum
      else:
        try:
          return_gen_coe_list = (repcoematrix * self.rep_to_gen_matrix().inv()).tolist()[0]
        except:
          return_gen_coe_list = []
      return return_gen_coe_list

    # def rep_coe_to_gen_coe(self, repcoelist):
    #   # print(f'repcoelist = {repcoelist}')
    #   repcoematrix = sp.Matrix([repcoelist])
    #   # print(f'rep_to_gen_matrix = {self.rep_to_gen_matrix()}')
    #   if repcoematrix == sp.Matrix([[0]]):
    #     direct_sum = self.direct_sum()
    #     return_gen_coe_list = [0] * direct_sum
    #   else:
    #     # print('aaaa')
    #     try:
    #       return_gen_coe_list = (repcoematrix * self.rep_to_gen_matrix().inv()).tolist()[0]
    #     except:
    #       return_gen_coe_list = []
    #   # if repcoelist != [] or repcoelist != [0]:
    #   #   print(f'repcoelist = {repcoelist}')
    #   #   repcoematrix = sp.Matrix([repcoelist])
    #   #   print(f'rep_to_gen_matrix = {self.rep_to_gen_matrix()}')
    #   #   if repcoematrix == sp.Matrix([[0]]):
    #   #     direct_sum = self.direct_sum()
    #   #     return_gen_coe_list = [0] * direct_sum
    #   #   else:
    #   #     print('aaaa')
    #   #     try:
    #   #       return_gen_coe_list = (repcoematrix * self.rep_to_gen_matrix().inv()).tolist()[0]
    #   #     except:
    #   #       return_gen_coe_list = []
    #   # else:
    #   #   return_gen_coe_list = []
    #   return return_gen_coe_list

    def gen_coe_to_rep_coe(self, gencoelist):
      if gencoelist != []:
        gencoematrix = sp.Matrix([gencoelist])
        return_rep_coe_list = (gencoematrix * self.rep_to_gen_matrix()).tolist()[0]
      else:
        return_rep_coe_list = []
      return return_rep_coe_list

    def mod_gen_coe_list(self, gencoe):
      if gencoe != []:
        order_list = self.order_list()
        direct_sum = self.direct_sum()
        def mod_coe(i):
          if order_list[i] == 0:
            return gencoe[i]
          elif order_list[i] == inf:
            return gencoe[i]
          elif gencoe[i] % order_list[i] > order_list[i] /2:
            return gencoe[i] % order_list[i] - order_list[i]
          else:
            return gencoe[i] % order_list[i]
        return_mod_gen_coe_list = [mod_coe(i) for i in range(direct_sum)]
      else:
        return_mod_gen_coe_list = []
      return return_mod_gen_coe_list

    # def P_coe(self, id):
    #   int_list = []
    #   ref_tex = ''
    #   order_list = self.order_list()
    #   try:
    #     if order_list == [0] or order_list == []:
    #       int_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #     else:
    #       queryid = f'select * from sphere where n = {self.n} and k = {self.k} and id = {id}'
    #       for row in c.execute(queryid):
    #         if row['P_coe'] is not None:
    #           int_list = list(map(int, row['P_coe'].split()))
    #           ref_tex = row['P']  
    #   except:
    #     pass
    #   del int_list[HomotopyGroup((self.n - 1) / 2, (self.n + 2 * self.k - 3) / 2).direct_sum():] 
    #   return int_list, ref_tex

    def gen_P_coe(self, id):
      int_list = []
      ref_tex = ''
      self_direct_sum = self.direct_sum()
      order_list = self.order_list()
      try:
        if order_list == [0] or order_list == []:
          int_list = [0] * 12
        else:
          queryid = f'select * from sphere where n = {self.n} and k = {self.k} and id = {id}'
          for row in c.execute(queryid):
            if row['P_coe'] is not None:
              int_list = list(map(int, row['P_coe'].split()))
              gen_int_list = list(map(int, row['gen_coe'].split()))
              del gen_int_list[self_direct_sum:]
              gen_matrix = sp.Matrix(gen_int_list)
              self_P_coe_matrix = self.P_coe_matrix()
              print(f'self_P_coe_matrix = {id, self_P_coe_matrix}')
              int_list = (gen_matrix.transpose()*self_P_coe_matrix).tolist()[0]
              int_list = list(map(int, row['P_coe'].split()))
              ref_tex = row['P']  
      except:
        pass
      del int_list[HomotopyGroup((self.n - 1) / 2, (self.n + 2 * self.k - 3) / 2).direct_sum():] 
      return int_list, ref_tex

    # def E_coe(self, id):
    #   int_list = []
    #   ref = []
    #   for row in c.execute(self.query_id(id)):
    #     if row['E_coe'] is not None:
    #       int_list = list(map(int, row['E_coe'].split()))
    #       ref = row['E'] #if row['E'] is not None else '' 
    #     hg = HomotopyGroup(self.n+1, self.k) if self.k+2 >= self.n \
    #       else HomotopyGroup(self.k+2, self.k)
    #     del int_list[hg.direct_sum():]
    #   return int_list, ref

    def gen_E_coe(self, id):
      ref = []
      self_direct_sum = self.direct_sum()
      for row in c.execute(self.query_id(id)):
        if row['E_coe'] is not None:
          gen_int_list = list(map(int, row['gen_coe'].split()))
          del gen_int_list[self_direct_sum:]
          gen_matrix = sp.Matrix(gen_int_list)
          self_E_coe_matrix = self.E_coe_matrix()
          return_int_list = (gen_matrix.transpose()*self_E_coe_matrix).tolist()[0]
          ref = row['E']
        else:
          return_int_list = []
        hg = HomotopyGroup(self.n+1, self.k) if self.k+2 >= self.n \
          else HomotopyGroup(self.k+2, self.k)
        del return_int_list[hg.direct_sum():]
      return return_int_list, ref

    # def gen_E_coe(self, id):
    #   # int_list = []
    #   ref = []
    #   self_direct_sum = self.direct_sum()
    #   for row in c.execute(self.query_id(id)):
    #     if row['E_coe'] is not None:
    #       # int_list = list(map(int, row['E_coe'].split())).
    #       # E_coe_list = list(map(int, row['E_coe'].split()))
    #       # if len(E_coe_list) == 0:
    #       #   print(f'bbbbb, {E_coe_list}')
          
    #       gen_int_list = list(map(int, row['gen_coe'].split()))
    #       del gen_int_list[self_direct_sum:]
    #       # print(f'gen_int_list = {gen_int_list}')
    #       gen_matrix = sp.Matrix(gen_int_list)
    #       self_E_coe_matrix = self.E_coe_matrix()
    #       print(f'gen_matrix = {gen_matrix.transpose()}')
    #       print(f'self_E_coe_matrix = {id, self_E_coe_matrix}')
    #       # print((gen_matrix.transpose()*self_E_coe_matrix).tolist())
    #       return_int_list = (gen_matrix.transpose()*self_E_coe_matrix).tolist()[0]
    #       ref = row['E'] #if row['E'] is not None else '' 
    #     else:
    #       return_int_list = []
    #     hg = HomotopyGroup(self.n+1, self.k) if self.k+2 >= self.n \
    #       else HomotopyGroup(self.k+2, self.k)
    #     # print(f'int_list = {int_list}')
    #     del return_int_list[hg.direct_sum():]
    #   # print(self.n+1, self.k, hg.direct_sum(), int_list)
    #   return return_int_list, ref

    # def H_coe(self, id):
    #   int_list = []
    #   ref_tex = ''
    #   if self.k + 2 >= self.n:
    #     for row in c.execute(self.query_id(id)):
    #       if row['H_coe'] is not None:
    #         int_list = list(map(int, row['H_coe'].split()))
    #         ref_tex = row['H']
    #   else:
    #     int_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #   del int_list[HomotopyGroup(2 * self.n - 1, self.k - self.n + 1).direct_sum():] 
    #   # リストの長さを direct_sum の個数にする
    #   return int_list, ref_tex

    def gen_H_coe(self, id):
      int_list = []
      ref_tex = ''
      self_direct_sum = self.direct_sum()
      if self.k + 2 >= self.n:
        for row in c.execute(self.query_id(id)):
          if row['H_coe'] is not None:
            int_list = list(map(int, row['H_coe'].split()))
            gen_int_list = list(map(int, row['gen_coe'].split()))
            del gen_int_list[self_direct_sum:]
            gen_matrix = sp.Matrix(gen_int_list)
            self_H_coe_matrix = self.H_coe_matrix()
            try:
              int_list = (gen_matrix.transpose()*self_H_coe_matrix).tolist()[0]
            except:
              int_list = [0]
            ref_tex = row['H']
      else:
        int_list = [0] * 12
      del int_list[HomotopyGroup(2 * self.n - 1, self.k - self.n + 1).direct_sum():] 
      return int_list, ref_tex

    # def mod_gen_coe_list(self, gencoe):
    #   def mod_coe(i):
    #     if self.order_list()[i] == inf:
    #       return gencoe[i]
    #     elif gencoe[i] % self.order_list()[i] > self.order_list()[i] /2:
    #       return gencoe[i] % self.order_list()[i] - self.order_list()[i]
    #     else:
    #       return gencoe[i] % self.order_list()[i]
    #   return_mod_gen_coe_list = [mod_coe(i) for i in range(self.direct_sum())]
    #   return return_mod_gen_coe_list

  nn = [n*2-1, n-1, n, n*2-1, n-1]
  kk = [k-n+2, k, k, k-n+1, k-1]
  print(f'n = {nn}')
  print(f'k = {kk}')
  HoGroup = []
  for i in range(5):
    txt_HoGroup = ' \pi_{ {{nn1 + kk1}} }^{ {{nn1}} } '
    tmp_HoGroup = Template(txt_HoGroup)
    dict_HoGroup = {'nn1':nn[i], 'kk1':kk[i]}
    HoGroup.append(tmp_HoGroup.render(dict_HoGroup))

  EHPmap = ['\Delta', 'E', 'H', '\Delta']
  Arrow = []
  for i in range(4):
    txt_Arrow = ' \stackrel{ {{map1}} }{\longrightarrow} '
    tmp_Arrow = Template(txt_Arrow)
    dict_Arrow = {'map1':EHPmap[i]}
    Arrow.append(tmp_Arrow.render(dict_Arrow))

  for i in range(5):
    if i==0:
      table = HoGroup[0]
    else:
      table = table + " & & & " + Arrow[i-1] + " & & & " + HoGroup[i]

  hg = [HomotopyGroup(nn[i], kk[i]) for i in range(5)]

  table_ref = [[], [], [], [], []]
  query = f'select * from sphere where n = {nn[0]} and k = {kk[0]}'
  table_ref[0] = [row['P'] for row in c.execute(query)]
  query = f'select * from sphere where n = {nn[1]} and k = {kk[1]}'
  table_ref[1] = [row['E'] for row in c.execute(query)]
  query = f'select * from sphere where n = {nn[2]} and k = {kk[2]}'
  table_ref[2] = [row['H'] for row in c.execute(query)]
  query = f'select * from sphere where n = {nn[3]} and k = {kk[3]}'
  table_ref[3] = [row['P'] for row in c.execute(query)]

  table_gen = [[], [], [], [], []]
  table_arrow = [[], [], [], [], []]
  table_image = [[], [], [], [], []]
  for i in range(5):
    for j in range(hg[i].direct_sum()):
      table_gen[i].append(hg[i].rep_linear_tex(hg[i].gen_coe_list(j)))
      if i < 4:
        table_arrow[i].append('\longrightarrow')
  
  # print(hg[2].H_coe(1)[0], hg[3].rep_coe_to_gen_coe(hg[2].H_coe(1)[0]))
  # print(hg[2].gen_H_coe(1)[0])
  # print('aaa')
  # print(hg[0].gen_P_coe(0)[0],hg[1].gen_E_coe(0)[0],hg[2].gen_H_coe(0)[0],hg[3].gen_P_coe(0)[0])

  # table_image[0] = [hg[1].rep_linear_tex(hg[0].gen_P_coe(j)[0]) 
  #   for j in range(hg[0].direct_sum())]
  # table_image[1] = [hg[2].rep_linear_tex(hg[2].rep_coe_to_gen_coe(hg[1].gen_E_coe(j)[0])) 
  #   for j in range(hg[1].direct_sum())]
  # table_image[2] = [hg[3].rep_linear_tex(hg[2].gen_H_coe(j)[0]) 
  #   for j in range(hg[2].direct_sum())]
  # table_image[3] = [hg[4].rep_linear_tex(hg[3].gen_P_coe(j)[0]) 
  #   for j in range(hg[3].direct_sum())]

  # print(hg[1].gen_E_coe(1)[0], hg[2].rep_coe_to_gen_coe(hg[1].gen_E_coe(1)[0]))
  table_image[0] = [hg[1].rep_linear_tex(hg[1].gen_coe_to_rep_coe(hg[1].mod_gen_coe_list(hg[1].rep_coe_to_gen_coe(hg[0].gen_P_coe(j)[0]))))
    for j in range(hg[0].direct_sum())]
  if nn[i] <= kk[i] + 2:
    table_image[1] = [hg[2].rep_linear_tex(hg[2].gen_coe_to_rep_coe(hg[2].mod_gen_coe_list(hg[2].rep_coe_to_gen_coe(hg[1].gen_E_coe(j)[0]))))
      for j in range(hg[1].direct_sum())]
  else:
    table_image[1] = table_gen[2]
  table_image[2] = [hg[3].rep_linear_tex(hg[3].gen_coe_to_rep_coe(hg[3].mod_gen_coe_list(hg[3].rep_coe_to_gen_coe(hg[2].gen_H_coe(j)[0]))))
    for j in range(hg[2].direct_sum())]
  table_image[3] = [hg[4].rep_linear_tex(hg[4].gen_coe_to_rep_coe(hg[4].mod_gen_coe_list(hg[4].rep_coe_to_gen_coe(hg[3].gen_P_coe(j)[0]))))
    for j in range(hg[3].direct_sum())]

  # table_image[0] = [hg[1].rep_linear_tex(hg[0].P_coe(j)[0]) 
  #   for j in range(hg[0].direct_sum())]
  # table_image[1] = [hg[2].rep_linear_tex(hg[1].E_coe(j)[0]) 
  #   for j in range(hg[1].direct_sum())]
  # table_image[2] = [hg[3].rep_linear_tex(hg[2].H_coe(j)[0]) 
  #   for j in range(hg[2].direct_sum())]
  # table_image[3] = [hg[4].rep_linear_tex(hg[3].P_coe(j)[0]) 
  #   for j in range(hg[3].direct_sum())]

  table_group = [[], [], [], [], []]
  for i in range(5):
    if nn[i] <= kk[i] + 2:
      query = f'select * from sphere where n = {nn[i]} and k = {kk[i]}'
    else:
      query = f'select * from sphere where n = {kk[i]+2} and k = {kk[i]}'
    for row in c.execute(query):
      if row['orders'] == 0:
        table_group[i].append('0')
      elif row['orders'] == inf:
        table_group[i].append('Z')
      else:
        try:
          orders = int(row['orders'])
          table_group[i].append(f'Z_{ {orders} }')
        except:
          table_group[i].append('')

  m_d_sum = hg[2].max_direct_sum()

  for i in range(5):
    if len(table_arrow[i]) < m_d_sum:
      for j in range(m_d_sum - len(table_arrow[i])):
        table_group[i].append('')
        table_gen[i].append('')
        table_arrow[i].append('')
        table_image[i].append('')
        table_ref[i].append('')

  conn.close()

  return render_template('homotopy_group.html', n=n, k=k, m_d_sum=m_d_sum \
    , HoGroup=HoGroup, Arrow=Arrow \
    , table_group=table_group, table_gen=table_gen \
    , table_arrow=table_arrow, table_image=table_image, table_ref=table_ref )


## おまじない
if __name__ == "__main__":
  app.run(debug=True)
