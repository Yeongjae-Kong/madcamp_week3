import React, { createContext, useMemo, useContext } from "react";
import { io } from "socket.io-client";

const SocketContext = createContext(null);

export const useSocket = () => {
  const socket = useContext(SocketContext);
  return socket;
};

export const SocketProvider = (props) => {
<<<<<<< HEAD
  const socket = useMemo(() => io("ws://3.15.28.55"), []);
=======
  const socket = useMemo(() => io("3.15.28.55"), []);
>>>>>>> 92030a1a5b086ddccf899c6bd734063c39dae48d

  return (
    <SocketContext.Provider value={socket}>
      {props.children}
    </SocketContext.Provider>
  );
};
