CC=clang++
TARGET=Release

INC=-I$(INTELOCLSDKROOT)include
CFLAGS=-c -Wall -std=c++11 $(INC)

ifeq ($(TARGET),Release)
# TODO: Clang 3.7.0 fails with -O2 on Windows
CFLAGS+= -O1
else
CFLAGS+= -O0
endif

SOURCES= 												\
	../implementation/data_ref.cpp						\
	../implementation/kernel_name.cpp					\
	../implementation/specification/buffer_base.cpp		\
	../implementation/specification/command_group.cpp	\
	../implementation/specification/context.cpp			\
	../implementation/specification/device.cpp			\
	../implementation/specification/device_selector.cpp	\
	../implementation/specification/error_handler.cpp	\
	../implementation/specification/event.cpp			\
	../implementation/specification/kernel.cpp			\
	../implementation/specification/platform.cpp		\
	../implementation/specification/program.cpp			\
	../implementation/specification/queue.cpp			\
	../implementation/specification/ranges/point.cpp	\
	../implementation/src_handlers/issue_command.cpp	\
	../implementation/src_handlers/kernel_source.cpp	\
	../implementation/synchronizer.cpp

OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=sycl-gtx-$(TARGET).a

ifeq ($(OS),Windows_NT)
RM=del
OBJECTS_RM=$(subst /,\, $(OBJECTS))
else
RM=rm
OBJECTS_RM=$(OBJECTS)
endif

all: $(SOURCES) $(EXECUTABLE)
	
# http://www.adp-gmbh.ch/cpp/gcc/create_lib.html
$(EXECUTABLE): $(OBJECTS)
	@echo --- Creating library: $@
	ar rcs $(EXECUTABLE) $(OBJECTS)

.cpp.o:
	@echo --- Building target: $@. First dependency is: $<
	$(CC) $(CFLAGS) $< -o $@

clean:
	$(RM) $(EXECUTABLE)
	$(RM) $(OBJECTS_RM)

test:
	@echo --- $(TARGET)
