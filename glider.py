import math

class Glider:
    def __init__(self, mass = 1.0, A = 1.0, phi = 0.0, x_dot = 10.0, y_dot = 0.0, y_init = 100.0, target_x = 10.0, delta_t = 0.01):
        self.gravity = 9.81
        self.rho = 1.225
        self.mass = mass
        self.A = A                
        self.x = 0.0
        self.x_dot = 10.0 # x_dot
        self.x_ddot = 0.0
        self.y_init = y_init
        self.y = y_init
        self.y_dot = y_dot
        self.y_ddot = 0.0
        self.phi = phi
        self.phi_dot = 0.0                
        self.delta_t = delta_t
        self.target_x = target_x
        self.lift = 0.0
        self.drag = 0.0
        
    def  reset(self):
        self.x = 0.0
        self.x_dot = 10.0
        self.y = self.y_init
        self.y_dot = 0.0
        self.phi = 0.0
        done = False
        return [self.x, self.x_dot, self.y, self.y_dot, self.phi, self.lift, self.drag]
        
    def Lift(self, phi, x_dot, y_dot):
        mass = 10.0
        rho = 1.225
        A = 1.0
        v_angle = math.atan(y_dot/x_dot)
        # print(v_angle)
        if (phi - v_angle) > (15/180*math.pi):
            C_L = 0.02*(1.75 - abs(phi - v_angle - 15/180*math.pi))
        else:
            C_L = 0.0
        return max(C_L, 0.0)*(x_dot**2 + y_dot**2)*rho*A/2

    def Drag(self, phi, x_dot, y_dot):
        rho = 1.225
        b = 1.0
        L = self.Lift(phi, x_dot, y_dot)
        D_i = L**2/(0.5*rho*math.pi*b)
        return D_i
 
    def probe_step(self, action):
        x_p = self.x + self.x_dot*self.delta_t + 0.5*self.x_ddot*self.delta_t**2
        x_dot_p = self.x_dot + self.x_ddot*self.delta_t
        y_p = self.y + self.y_dot*self.delta_t + 0.5*self.y_ddot*self.delta_t**2
        y_dot_p = self.y_dot + self.y_ddot*self.delta_t
        phi_p = max(min(math.pi/2, self.phi + 10*self.delta_t*action), -math.pi/2)
        return [x_p, x_dot_p, y_p, y_dot_p, phi_p]
    
    def probe_step_state(self, state_p, action):
        [x_p, x_dot_p, x_ddot_p, y_p, y_dot_p, y_ddot_p, phi_p] = state_p
        x_p = x_p + x_dot_p*self.delta_t + 0.5*x_ddot_p*self.delta_t**2
        x_dot_p = x_dot_p + x_ddot_p*self.delta_t
        y_p = y_p + y_dot_p*self.delta_t + 0.5*y_ddot_p*self.delta_t**2
        y_dot_p = y_dot_p + y_ddot_p*self.delta_t
        phi_p = max(min(math.pi/2, phi_p + 10*self.delta_t*action), -math.pi/2)
        return [x_p, x_dot_p, x_ddot_p, y_p, y_dot_p, y_ddot_p, phi_p]
    
    def step(self, action):
        eps = 1e-6
        done = False
        # reward = -abs(self.h - self.h_init*(self.target_x-self.x)/self.target_x)
        x0 = self.x
        y0 = self.y
        self.lift = self.Lift(self.phi, self.x_dot, self.y_dot)
        self.drag = self.Drag(self.phi, self.x_dot, self.y_dot)
        
        [self.x, self.x_dot, self.y, self.y_dot, self.phi] = self.probe_step(action)
        # self.x = self.x + self.x_dot*self.delta_t + 0.5*self.x_ddot*self.delta_t**2
        # self.x_dot = self.x_dot + self.x_ddot*self.delta_t 
        D_pX = 0.000001*(self.x_dot**2)
        self.x_ddot = 1/self.mass*(self.lift*math.cos(self.phi) - self.drag*math.sin(self.phi) - D_pX)
                
        # self.y = self.y + self.y_dot*self.delta_t + 0.5*self.y_ddot*self.delta_t**2
        # self.y_dot = self.y_dot + self.y_ddot*self.delta_t   
        D_py = 0.000001*(self.y_dot**2)
        self.y_ddot = 1/self.mass*(self.lift*math.cos(self.phi) - self.drag*math.sin(self.phi) - D_py) - self.gravity         

        # self.phi = max(min(math.pi/2, self.phi + 10*self.delta_t*action), -math.pi/2)
        self.phi_dot = 0.0    
        
        self.lift = self.Lift(self.phi, self.x_dot, self.y_dot)
        self.drag = self.Drag(self.phi, self.x_dot, self.y_dot)
        
        if self.y <= 0.0:
            self.x = x0 + y0/(y0-self.y)*(self.x - x0)
            self.y = 0.0
            done = True
            
        if action == 0.0:
            cost = 0.0
        else:
            cost = 1.0
            
        return cost, done
    
    def probe_n_step(self, action, n_steps, opt_update):
        eps = 1e-6
        done = False
        # reward = -abs(self.h - self.h_init*(self.target_x-self.x)/self.target_x)
        x0 = self.x
        y0 = self.y
        
        x_p = self.x
        x_dot_p = self.x_dot
        x_ddot_p = self.x_ddot
        y_p = self.y
        y_dot_p = self.y_dot
        y_ddot_p = self.y_ddot
        phi_p = self.phi
        
        for i in range(n_steps):            
            n_lift = self.Lift(self.phi, self.x_dot, self.y_dot)
            n_drag = self.Drag(self.phi, self.x_dot, self.y_dot)

            [x_p, x_dot_p, x_ddot_p, y_p, y_dot_p, y_ddot_p, phi_p] = self.probe_step_state([x_p, x_dot_p, x_ddot_p, y_p, y_dot_p, y_ddot_p, phi_p], action)
            # self.x = self.x + self.x_dot*self.delta_t + 0.5*self.x_ddot*self.delta_t**2
            # self.x_dot = self.x_dot + self.x_ddot*self.delta_t 
            D_pX = 0.000001*(x_dot_p**2)
            x_ddot_p = 1/self.mass*(n_lift*math.cos(phi_p) - n_drag*math.sin(phi_p) - D_pX)

            # self.y = self.y + self.y_dot*self.delta_t + 0.5*self.y_ddot*self.delta_t**2
            # self.y_dot = self.y_dot + self.y_ddot*self.delta_t   
            D_py = 0.000001*(y_dot_p**2)
            y_ddot_p = 1/self.mass*(n_lift*math.cos(phi_p) - n_drag*math.sin(phi_p) - D_py) - self.gravity         

            # self.phi = max(min(math.pi/2, self.phi + 10*self.delta_t*action), -math.pi/2)
            # phi_dot = 0.0    
        
        if not opt_update:
            return [x_p, x_dot_p, y_p, y_dot_p, phi_p]
        
        self.x = x_p
        self.x_dot = x_dot_p
        self.x_ddot = x_ddot_p
        self.y = y_p
        self.y_dot = y_dot_p
        self.y_ddot = y_ddot_p
        self.phi = phi_p
        self.lift = self.Lift(self.phi, self.x_dot, self.y_dot)
        self.drag = self.Drag(self.phi, self.x_dot, self.y_dot)                

        if self.y <= 0.0:
            self.x = x0 + y0/(y0-self.y)*(self.x - x0)
            self.y = 0.0
            done = True
            
        if action == 0.0:
            cost = 0.0
        else:
            cost = 1.0
            
        return cost, done
    
    def get_state(self):
        return [self.x, self.x_dot, self.y, self.y_dot, self.phi, self.lift, self.drag]
        